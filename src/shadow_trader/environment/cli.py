"""``shadow-trader env`` command group.

``env regen-golden`` is the only way to refresh the golden hash and Ledger tables
checked in beside the fixture Dataset. It refuses a dirty git tree (the golden must
come from committed code) and a run on any platform but ``GOLDEN_PLATFORM`` (the
hash is recorded with its platform). Tests never invoke the regeneration itself.

Exit codes: 0 success, 1 refused or failed, 2 setup error (not run from a source
checkout). Errors go to stderr; stdout carries only the result line.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from .determinism import GOLDEN_PLATFORM, current_platform

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_SETUP_ERROR = 2

SCENARIO_MODULE = "tests.environment.golden"


def add_env_parser(subparsers: argparse._SubParsersAction) -> None:
    env = subparsers.add_parser("env", help="Trading environment tools.")
    commands = env.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser(
        "regen-golden",
        help="Regenerate the golden hash and Ledger tables.",
        description=(
            "Replay the golden scenario and rewrite the golden files. Refuses a "
            f"dirty git tree and any platform but {GOLDEN_PLATFORM}."
        ),
    )


def tree_is_dirty() -> bool:
    """Whether the git working tree has uncommitted or untracked changes."""
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return bool(out.stdout.strip())


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def run_env(args: argparse.Namespace) -> int:
    if args.command == "regen-golden":
        return _regen_golden()
    raise AssertionError(args.command)  # argparse requires a known command


def _regen_golden() -> int:
    if tree_is_dirty():
        sys.stderr.write("refusing to regenerate: the git tree is dirty\n")
        return EXIT_REFUSED
    platform = current_platform()
    if platform != GOLDEN_PLATFORM:
        sys.stderr.write(
            f"refusing to regenerate: this platform is {platform}, the golden is "
            f"recorded on {GOLDEN_PLATFORM}\n"
        )
        return EXIT_REFUSED
    try:
        root = _repo_root()
    except (OSError, subprocess.CalledProcessError):
        sys.stderr.write("regen-golden must run inside the source checkout\n")
        return EXIT_SETUP_ERROR
    sys.path.insert(0, str(root))
    try:
        scenario = importlib.import_module(SCENARIO_MODULE)
    except ImportError:
        sys.stderr.write(f"cannot import {SCENARIO_MODULE} from {root}\n")
        return EXIT_SETUP_ERROR
    digest = scenario.write_golden(platform)
    sys.stdout.write(f"golden {digest} ({platform})\n")
    return EXIT_OK
