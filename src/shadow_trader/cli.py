"""Command-line entry point. Subcommands take no flags: config lives in `config`."""

import argparse
from collections.abc import Sequence

from shadow_trader import fetch

SUBCOMMANDS = {
    "fetch-data": "Fetch and prepare the EUR/USD M5 history.",
    "train": "Train the PPO agent.",
    "evaluate": "Evaluate the trained agent against the baselines.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shadow-trader")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in SUBCOMMANDS.items():
        sub.add_parser(name, help=text, description=text)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "fetch-data":
        fetch.run()
        return
    print(f"shadow-trader {args.command}: not yet implemented")
