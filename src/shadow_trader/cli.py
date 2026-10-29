"""Command-line entry point. Subcommands take no flags: config lives in `config`."""

import argparse
from collections.abc import Callable, Sequence

from shadow_trader import bars, evaluation, fetch, training

SUBCOMMANDS = {
    "fetch-data": "Fetch and prepare the EUR/USD M5 history.",
    "train": "Train the PPO Agent.",
    "evaluate": "Evaluate the trained Agent against the baselines.",
}


def _fetch_data() -> None:
    result = fetch.run()
    bars.ingest(result.weeks, bars.bars_path())


HANDLERS: dict[str, Callable[[], None]] = {
    "fetch-data": _fetch_data,
    "train": training.run,
    "evaluate": evaluation.run,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shadow-trader")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in SUBCOMMANDS.items():
        sub.add_parser(name, help=text, description=text)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    HANDLERS[args.command]()
