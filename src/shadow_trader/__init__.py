import argparse
from collections.abc import Sequence

from shadow_trader.environment.cli import add_env_parser, run_env
from shadow_trader.ingestion.cli import add_ingest_parser, run_ingest
from shadow_trader.preparation.cli import add_prepare_parser, run_prepare


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow-trader")
    subparsers = parser.add_subparsers(dest="subsystem", required=True)
    add_ingest_parser(subparsers)
    add_prepare_parser(subparsers)
    add_env_parser(subparsers)
    args = parser.parse_args(argv)
    if args.subsystem == "prepare":
        return run_prepare(args)
    if args.subsystem == "env":
        return run_env(args)
    return run_ingest(args)


__all__ = ["main"]
