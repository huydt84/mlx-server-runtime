"""Tests for the shared host-validation command-line parser."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "mlx-host-validation/scripts"
CLI_LIB = SCRIPTS_DIR / "lib/cli.sh"
HOST_SCRIPTS = sorted(SCRIPTS_DIR.glob("*.sh"))

DEMO_SCRIPT = """
set -euo pipefail
source "$CLI_LIB"
cli_describe "Demo script."
cli_option PORT --port DEMO_PORT port 18190 "Gateway port."
cli_option CHECKPOINT --checkpoint DEMO_CHECKPOINT string "org/model" "Checkpoint."
cli_option CAPTURE --metal-capture DEMO_CAPTURE bool 0 "Capture."
cli_option XCTRACE --xctrace DEMO_XCTRACE bool "" "Derived from capture."
cli_option RATIO --max-regression DEMO_RATIO number 0.02 "Ratio."
cli_option SAMPLES --samples DEMO_SAMPLES int 10 "Samples."
cli_option ITEMS --items DEMO_ITEMS list "a,b" "Items."
cli_parse "$@"
XCTRACE="${XCTRACE:-$CAPTURE}"
cli_split_list ITEM_ARRAY "$ITEMS"
echo "xctrace=$XCTRACE"
echo "item_count=${#ITEM_ARRAY[@]}"
"""


def _env(**overrides: str) -> dict[str, str]:
    """Return a clean environment without the demo variables."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("DEMO_")
    }
    env["CLI_LIB"] = str(CLI_LIB)
    env.update(overrides)
    return env


def _demo(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the demo script under bash with the given arguments."""
    return subprocess.run(
        ["bash", "-c", DEMO_SCRIPT, "demo", *args],
        capture_output=True,
        text=True,
        env=_env(**env),
        check=False,
    )


def _options(stdout: str) -> dict[str, str]:
    """Collect the `key=value` lines printed by the script."""
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def test_defaults_resolve_and_print():
    """Unset options take their defaults and are printed as option_* lines."""
    result = _demo()
    assert result.returncode == 0, result.stderr
    values = _options(result.stdout)
    assert values["option_port"] == "18190"
    assert values["option_checkpoint"] == "org/model"
    assert values["option_metal_capture"] == "0"
    assert values["option_max_regression"] == "0.02"
    assert values["xctrace"] == "0"
    assert values["item_count"] == "2"


def test_flag_overrides_environment_which_overrides_default():
    """A flag beats the environment variable, which beats the default."""
    from_env = _options(_demo(DEMO_PORT="18300").stdout)
    assert from_env["option_port"] == "18300"

    from_flag = _options(_demo("--port", "18400", DEMO_PORT="18300").stdout)
    assert from_flag["option_port"] == "18400"

    equals_form = _options(_demo("--port=18500", DEMO_PORT="18300").stdout)
    assert equals_form["option_port"] == "18500"


def test_bool_flags_and_derived_defaults():
    """Bool flags take --flag/--no-flag and an empty default stays derivable."""
    enabled = _options(_demo("--metal-capture").stdout)
    assert enabled["option_metal_capture"] == "1"
    assert enabled["xctrace"] == "1"

    disabled = _options(_demo("--no-metal-capture", DEMO_CAPTURE="1").stdout)
    assert disabled["option_metal_capture"] == "0"

    explicit = _options(_demo("--metal-capture", "--no-xctrace").stdout)
    assert explicit["option_xctrace"] == "0"
    assert explicit["xctrace"] == "0"


def test_list_option_splits_on_commas():
    """A list option splits into a bash array on commas."""
    values = _options(_demo("--items", "x,y,z").stdout)
    assert values["option_items"] == "x,y,z"
    assert values["item_count"] == "3"


def test_print_config_exits_before_the_script_body():
    """--print-config prints the resolved options and stops the script."""
    result = _demo("--print-config", "--samples", "4")
    assert result.returncode == 0
    assert "option_samples=4" in result.stdout
    assert not any(line.startswith("xctrace=") for line in result.stdout.splitlines())


def test_help_lists_flags_environment_and_defaults():
    """--help lists every flag with its environment variable and default."""
    result = _demo("--help")
    assert result.returncode == 0
    assert "Demo script." in result.stdout
    assert "--port PORT" in result.stdout
    assert "env DEMO_PORT; default: 18190" in result.stdout
    assert "--metal-capture, --no-metal-capture" in result.stdout
    assert "default: (derived)" in result.stdout


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("--port", "70000"), {}, "--port must be a TCP port in 1..65535"),
        (("--port", "abc"), {}, "--port must be a TCP port"),
        (("--port",), {}, "--port requires a value"),
        (("--samples", "-1"), {}, "--samples must be a non-negative integer"),
        (
            ("--max-regression", "1e3"),
            {},
            "--max-regression must be a non-negative number",
        ),
        ((), {"DEMO_CAPTURE": "yes"}, "--metal-capture must be 0 or 1"),
        (("--bogus",), {}, "unknown option --bogus"),
        (("--no-port",), {}, "unknown option --no-port"),
        (("stray",), {}, "unexpected argument 'stray'"),
    ],
)
def test_usage_errors_exit_two(args, env, message):
    """Invalid values, unknown flags, and stray arguments exit 2 with a message."""
    result = _demo(*args, **env)
    assert result.returncode == 2
    assert message in result.stderr


def test_require_free_port_names_the_flag_for_a_busy_port():
    """A busy port fails fast and names the flag that changes it."""
    script = 'set -euo pipefail; source "$CLI_LIB"; cli_require_free_port "$1" --native-port; echo free'
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = str(listener.getsockname()[1])
        busy = subprocess.run(
            ["bash", "-c", script, "demo", port],
            capture_output=True,
            text=True,
            env=_env(),
            check=False,
        )
    if busy.returncode == 0:
        pytest.skip("lsof is unavailable or cannot see local listeners")
    assert busy.returncode == 1
    assert f"port {port} is already in use" in busy.stderr
    assert "--native-port" in busy.stderr

    free = subprocess.run(
        ["bash", "-c", script, "demo", port],
        capture_output=True,
        text=True,
        env=_env(),
        check=False,
    )
    assert free.returncode == 0, free.stderr
    assert free.stdout.strip() == "free"


@pytest.mark.parametrize("script", HOST_SCRIPTS, ids=lambda path: path.name)
def test_every_host_script_parses_arguments_before_side_effects(script: Path):
    """Every host script answers --print-config, --help, and unknown flags without running."""
    printed = subprocess.run(
        ["bash", str(script), "--print-config"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert printed.returncode == 0, printed.stderr
    assert all(line.startswith("option_") for line in printed.stdout.splitlines())

    rejected = subprocess.run(
        ["bash", str(script), "--definitely-not-an-option"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "unknown option --definitely-not-an-option" in rejected.stderr

    helped = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert helped.returncode == 0
    assert helped.stdout.startswith("Usage: bash ")
    assert "--print-config" in helped.stdout


def test_every_gateway_port_is_configurable():
    """No host script hardcodes the default gateway port 8000."""
    for script in HOST_SCRIPTS:
        text = script.read_text()
        assert "127.0.0.1:8000" not in text, script.name
        assert "TCP:8000" not in text, script.name
