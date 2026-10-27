import subprocess
import sys
from pathlib import Path

import pytest

SUBCOMMANDS = ["fetch-data", "train", "evaluate"]
ENTRY_POINT = str(Path(sys.executable).parent / "shadow-trader")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ENTRY_POINT, *args], capture_output=True, text=True, check=False
    )


def test_help_lists_exactly_the_three_subcommands():
    result = run("--help")
    assert result.returncode == 0
    listed = result.stdout.split("{")[1].split("}")[0].split(",")
    assert listed == SUBCOMMANDS


@pytest.mark.parametrize("name", ["train", "evaluate"])
def test_subcommand_exits_cleanly_with_notice(name):
    result = run(name)
    assert result.returncode == 0
    assert "not yet implemented" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_accepts_no_override_flags(name):
    assert run(name, "--seed", "1").returncode != 0
    options = [line for line in run(name, "--help").stdout.splitlines() if "--" in line]
    assert all("--help" in line for line in options)
