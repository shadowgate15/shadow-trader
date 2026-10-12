"""``shadow-trader ingest`` command group.

Exit codes: 0 success (warnings included), 1 build failure, 2 config error.
Flags may override only instruments, time range and bar sizes, never rule
values; argparse rejects any other flag with exit code 2.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shadow_trader.ingestion.config import ConfigError, load_config

EXIT_OK = 0
EXIT_BUILD_FAILURE = 1
EXIT_CONFIG_ERROR = 2

_COMMANDS = {
    "fetch": "Fetch raw months into the cache (the only network command).",
    "build": "Build a backfill Dataset from the raw cache.",
    "live": "Run the live ingester.",
    "build-live": "Build a live Dataset over sealed slots.",
    "diff": "Advisory live-vs-backfill reconciliation.",
}


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _int_csv(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in _csv(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a list of integers: {value!r}") from None


def add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    ingest = subparsers.add_parser("ingest", help="Market data ingestion.")
    commands = ingest.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, help_text in _COMMANDS.items():
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument(
            "--config", type=Path, required=True, help="Path to the TOML config."
        )
        command.add_argument("--instruments", type=_csv, help="Override instruments.")
        command.add_argument("--start", help="Override the first UTC month (YYYY-MM).")
        command.add_argument("--end", help="Override the last UTC month (YYYY-MM).")
        command.add_argument(
            "--bar-sizes", type=_int_csv, help="Override bar sizes in minutes."
        )
        if name == "build-live":
            command.add_argument("--from", dest="range_from", help="Range start.")
            command.add_argument("--to", dest="range_to", help="Range end.")


def run_ingest(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config).with_overrides(
            instruments=args.instruments,
            bar_sizes=args.bar_sizes,
            start=args.start,
            end=args.end,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    del config  # consumed by the real commands in later tickets
    print(f"ingest {args.command}: not implemented yet", file=sys.stderr)
    return EXIT_BUILD_FAILURE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow-trader")
    subparsers = parser.add_subparsers(dest="subsystem", required=True)
    add_ingest_parser(subparsers)
    args = parser.parse_args(argv)
    return run_ingest(args)
