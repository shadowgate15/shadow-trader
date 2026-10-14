"""``shadow-trader ingest`` command group.

Exit codes: 0 success (warnings included), 1 build failure, 2 config error,
130 interrupted (Ctrl-C).
Progress (phases, bars, tolerated warnings) goes to stderr through ``rich``;
stdout carries only the final result lines. ``-q`` silences progress, ``-v``
adds per-month detail.
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

from rich.console import Console

from .backfill import BuildFailure, build_backfill
from .bars import Bar
from .cli_progress import NORMAL, QUIET, VERBOSE, RichProgress
from .config import Config, ConfigError, load_config
from .dataset import DatasetError, RawCache
from .live import LiveIngester
from .livedataset import build_live
from .reconcile import ReconciliationRefused, reconcile
from .replay import live_adapter
from .report import console_summary
from .sources import (
    Transport,
    UrllibTransport,
    fetch_months,
)

EXIT_OK = 0
EXIT_BUILD_FAILURE = 1
EXIT_CONFIG_ERROR = 2
EXIT_INTERRUPTED = 130

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
    _add_verbosity(ingest, default=False)
    commands = ingest.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, help_text in _COMMANDS.items():
        command = commands.add_parser(name, help=help_text, description=help_text)
        _add_verbosity(command, default=argparse.SUPPRESS)
        command.add_argument(
            "--config", type=Path, required=True, help="Path to the TOML config."
        )
        command.add_argument("--instruments", type=_csv, help="Override instruments.")
        command.add_argument("--start", help="Override the first UTC month (YYYY-MM).")
        command.add_argument("--end", help="Override the last UTC month (YYYY-MM).")
        command.add_argument(
            "--bar-sizes", type=_int_csv, help="Override bar sizes in minutes."
        )
        if name == "fetch":
            command.add_argument(
                "--workers",
                type=int,
                help="Concurrent fetch workers (positive; default from config, 8).",
            )
        if name == "build-live":
            command.add_argument("--from", dest="range_from", help="Range start.")
            command.add_argument("--to", dest="range_to", help="Range end.")
        if name == "diff":
            command.add_argument("--live", dest="live_id", help="Live dataset_id.")
            command.add_argument(
                "--backfill", dest="backfill_id", help="Backfill dataset_id."
            )


def _add_verbosity(parser: argparse.ArgumentParser, default: object) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=default,
        help="No progress: only the final result lines and errors.",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default,
        help="Per-month detail, including every tolerated warning.",
    )


def make_console() -> Console:
    """The progress console: stderr, so stdout keeps only the result lines."""
    return Console(stderr=True)


def _fail(progress: RichProgress, message: str) -> None:
    """Stop the live display, then print an error naming the phase it hit."""
    progress.close()
    where = f" during {progress.phase}" if progress.phase else ""
    print(message.replace("{where}", where), file=sys.stderr)


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


def _run_fetch(config: Config, args: argparse.Namespace, progress: RichProgress) -> int:
    report = fetch_months(
        RawCache(config.paths.data_root),
        make_transport(),
        config.build.instruments,
        config.build.start,
        config.build.end,
        sleep=_sleep,
        progress=progress,
        workers=config.build.workers,
    )
    progress.close()
    print(
        f"fetched {len(report.fetched)}, skipped {len(report.skipped)} cached, "
        f"absent {len(report.absent)}, unsettled {len(report.unsettled)}, "
        f"rejected {len(report.rejected)}"
    )
    for instrument, period in report.unsettled:
        print(f"warning: {instrument} {period} not settled yet; fetch it later")
    for instrument, period in report.absent:
        print(f"warning: {instrument} {period} absent at source")
    for (instrument, period), reason in report.rejected:
        print(f"error: {instrument} {period} rejected: {reason}", file=sys.stderr)
    if report.interrupted:
        print("interrupted: partial fetch; re-run to fetch the missing months")
        return EXIT_INTERRUPTED
    return EXIT_OK if report.ok else EXIT_BUILD_FAILURE


def _run_build(config: Config, args: argparse.Namespace, progress: RichProgress) -> int:
    try:
        for bar_size in config.build.bar_sizes:
            result = build_backfill(config, bar_size, progress=progress)
            progress.close()
            print(f"built {bar_size} min dataset {result.ref.dataset_id}")
            assert result.report_path is not None
            report = json.loads(result.report_path.read_text())
            for line in console_summary(report, result.report_path):
                print(line)
    except BuildFailure as exc:
        at = f" at {exc.locator}" if exc.locator else ""
        _fail(progress, f"build failed{{where}}{at}")
        if exc.report_path is not None:
            report = json.loads(exc.report_path.read_text())
            for line in console_summary(report, exc.report_path):
                print(line, file=sys.stderr)
        else:
            print(f"build failed: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILURE
    return EXIT_OK


def _run_build_live(
    config: Config, args: argparse.Namespace, progress: RichProgress
) -> int:
    start = _parse_utc(args.range_from, "--from")
    end = _parse_utc(args.range_to, "--to")
    if end <= start:
        raise ConfigError(f"--from {start.isoformat()} is not before --to")
    try:
        for bar_size in config.build.bar_sizes:
            ref = build_live(config, bar_size, start, end, progress=progress)
            progress.close()
            print(f"built live {bar_size} min dataset {ref.dataset_id}")
    except (DatasetError, OSError) as exc:
        _fail(progress, f"build-live failed{{where}}: {exc}")
        return EXIT_BUILD_FAILURE
    return EXIT_OK


def _run_diff(config: Config, args: argparse.Namespace, progress: RichProgress) -> int:
    if not (args.live_id and args.backfill_id):
        raise ConfigError("diff needs --live and --backfill dataset ids")
    try:
        path = reconcile(config, args.live_id, args.backfill_id, progress=progress)
        progress.close()
    except ReconciliationRefused as exc:
        print(f"diff refused: {exc}", file=sys.stderr)
        return EXIT_BUILD_FAILURE
    print(f"report: {path}")
    return EXIT_OK


def _run_live(config: Config, args: argparse.Namespace, progress: RichProgress) -> int:
    if len(config.build.bar_sizes) != 1:
        raise ConfigError(
            "live runs one bar size; pass --bar-sizes N "
            f"(configured: {', '.join(map(str, config.build.bar_sizes))})"
        )
    try:
        for instrument in config.build.instruments:
            bars = 0

            def count(bar: Bar, revision: int) -> None:
                nonlocal bars
                bars += 1

            LiveIngester(
                config,
                live_adapter(config, instrument),
                sink=count,
                progress=progress,
            ).run()
            progress.close()
            print(f"live {instrument}: {bars} bar emissions")
    except (BuildFailure, OSError) as exc:
        _fail(progress, f"live failed{{where}}: {exc}")
        return EXIT_BUILD_FAILURE
    return EXIT_OK


_HANDLERS: dict[str, Callable[[Config, argparse.Namespace, RichProgress], int]] = {
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
            workers=getattr(args, "workers", None),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    level = QUIET if args.quiet else VERBOSE if args.verbose else NORMAL
    with RichProgress(make_console(), level) as progress:
        try:
            return _HANDLERS[args.command](config, args, progress)
        except ConfigError as exc:
            _fail(progress, f"config error: {exc}")
            return EXIT_CONFIG_ERROR
        except KeyboardInterrupt:
            _fail(progress, "interrupted{where}")
            return EXIT_INTERRUPTED


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow-trader")
    subparsers = parser.add_subparsers(dest="subsystem", required=True)
    add_ingest_parser(subparsers)
    args = parser.parse_args(argv)
    return run_ingest(args)
