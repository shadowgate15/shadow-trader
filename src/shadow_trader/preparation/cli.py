"""``shadow-trader prepare`` command group.

Exit codes: 0 success, 1 build failure, 2 config error. Progress goes to stderr
through ``rich``; stdout carries only the result line. The only flag that
selects anything is ``--config``: every build setting lives in the config file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from shadow_trader.ingestion.cli_progress import NORMAL, QUIET, VERBOSE, RichProgress
from shadow_trader.ingestion.config import ConfigError

from .build import BuildFailure, build_prepared
from .config import load_prepared_config
from .verify import verify_build

EXIT_OK = 0
EXIT_BUILD_FAILURE = 1
EXIT_CONFIG_ERROR = 2


def add_prepare_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    prepare = subparsers.add_parser("prepare", help="Dataset preparation.")
    commands = prepare.add_subparsers(dest="command", required=True, metavar="COMMAND")
    build = commands.add_parser(
        "build",
        help="Build a prepared Dataset from an ingestion Dataset.",
        description="Build a prepared Dataset from an ingestion Dataset.",
    )
    build.add_argument(
        "--config", type=Path, required=True, help="Path to the TOML config."
    )
    group = build.add_mutually_exclusive_group()
    group.add_argument("-q", "--quiet", action="store_true", help="No progress.")
    group.add_argument("-v", "--verbose", action="store_true", help="More detail.")
    verify = commands.add_parser(
        "verify",
        help="Audit a stored build without writing anything.",
        description="Rehash Datasets, rerun boundary_check, resolve the scaler.",
    )
    verify.add_argument("build_id", help="The build to audit.")
    verify.add_argument(
        "--config", type=Path, required=True, help="Path to the TOML config."
    )


def run_verify(args: argparse.Namespace) -> int:
    try:
        config = load_prepared_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    checks = verify_build(config, args.build_id)
    for check in checks:
        print(
            f"PASS {check.name}"
            if check.passed
            else f"FAIL {check.name}: {check.detail}"
        )
    return EXIT_OK if all(c.passed for c in checks) else EXIT_BUILD_FAILURE


def run_prepare(args: argparse.Namespace) -> int:
    if args.command == "verify":
        return run_verify(args)
    try:
        config = load_prepared_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    level = QUIET if args.quiet else VERBOSE if args.verbose else NORMAL
    with RichProgress(Console(stderr=True), level) as progress:
        try:
            ref = build_prepared(config, progress=progress)
        except BuildFailure as exc:
            progress.close()
            print(f"build failed: {exc}", file=sys.stderr)
            return EXIT_BUILD_FAILURE
        progress.close()
    print(f"built prepared build {ref.build_id}")
    for role, dataset in ref.datasets.items():
        print(f"{role} {dataset.dataset_id}")
    return EXIT_OK
