"""``shadow-trader ingest`` command group.

Exit codes: 0 success (warnings included), 1 build failure, 2 config error.
Flags may override only instruments, time range and bar sizes, never rule
values; argparse rejects any other flag with exit code 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .backfill import BuildFailure, build_backfill
from .config import Config, ConfigError, load_config
from .dataset import DatasetError, RawCache
from .livedataset import build_live
from .reconcile import ReconciliationRefused, reconcile
from .report import console_summary
from .sources import (
    Transport,
    UrllibTransport,
    fetch_months,
)

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
        if name == "diff":
            command.add_argument("--live", dest="live_id", help="Live dataset_id.")
            command.add_argument(
                "--backfill", dest="backfill_id", help="Backfill dataset_id."
            )


def _parse_utc(value: str | None, flag: str) -> datetime:
    if value is None:
        raise ConfigError(f"{flag} is required")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ConfigError(
            f"{flag} must be an ISO date or datetime: {value!r}"
        ) from None
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def make_transport() -> Transport:
    return UrllibTransport()


_sleep = time.sleep


def _run_fetch(config: Config, args: argparse.Namespace) -> int:
    report = fetch_months(
        RawCache(config.paths.data_root),
        make_transport(),
        config.build.instruments,
        config.build.start,
        config.build.end,
        sleep=_sleep,
    )
    print(
        f"fetched {len(report.fetched)}, skipped {len(report.skipped)} cached, "
        f"absent {len(report.absent)}, rejected {len(report.rejected)}"
    )
    for instrument, period in report.absent:
        print(f"warning: {instrument} {period} absent at source")
    for (instrument, period), reason in report.rejected:
        print(f"error: {instrument} {period} rejected: {reason}", file=sys.stderr)
    return EXIT_OK if report.ok else EXIT_BUILD_FAILURE


def _run_build(config: Config, args: argparse.Namespace) -> int:
    try:
        for bar_size in config.build.bar_sizes:
            result = build_backfill(config, bar_size)
            print(f"built {bar_size} min dataset {result.ref.dataset_id}")
            assert result.report_path is not None
            report = json.loads(result.report_path.read_text())
            for line in console_summary(report, result.report_path):
                print(line)
    except BuildFailure as exc:
        if exc.report_path is not None:
            report = json.loads(exc.report_path.read_text())
            for line in console_summary(report, exc.report_path):
                print(line, file=sys.stderr)
        else:
            print(f"build failed: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILURE
    return EXIT_OK


def _run_build_live(config: Config, args: argparse.Namespace) -> int:
    start = _parse_utc(args.range_from, "--from")
    end = _parse_utc(args.range_to, "--to")
    if end <= start:
        raise ConfigError(f"--from {start.isoformat()} is not before --to")
    try:
        for bar_size in config.build.bar_sizes:
            ref = build_live(config, bar_size, start, end)
            print(f"built live {bar_size} min dataset {ref.dataset_id}")
    except (DatasetError, OSError) as exc:
        print(f"build-live failed: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILURE
    return EXIT_OK


def _run_diff(config: Config, args: argparse.Namespace) -> int:
    if not (args.live_id and args.backfill_id):
        raise ConfigError("diff needs --live and --backfill dataset ids")
    try:
        path = reconcile(config, args.live_id, args.backfill_id)
    except ReconciliationRefused as exc:
        print(f"diff refused: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILURE
    print(f"report: {path}")
    return EXIT_OK


def _run_live(config: Config, args: argparse.Namespace) -> int:
    print("ingest live: not implemented yet", file=sys.stderr)
    return EXIT_BUILD_FAILURE


_HANDLERS: dict[str, Callable[[Config, argparse.Namespace], int]] = {
    "fetch": _run_fetch,
    "build": _run_build,
    "live": _run_live,
    "build-live": _run_build_live,
    "diff": _run_diff,
}


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

    try:
        return _HANDLERS[args.command](config, args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow-trader")
    subparsers = parser.add_subparsers(dest="subsystem", required=True)
    add_ingest_parser(subparsers)
    args = parser.parse_args(argv)
    return run_ingest(args)
