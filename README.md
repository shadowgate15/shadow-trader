# Shadow Trader

A reinforcement-learning forex trading bot, in MVP form. It learns a trading policy for EUR/USD from historical five-minute prices and is judged offline, on a held-out test period, against two simple baselines (buy-and-hold and a turnover-matched random policy).

The MVP is three subcommands:

1. `fetch-data` downloads the pinned EUR/USD History once, aggregates it to M5 Bars carrying both bid and ask sides, validates it, and prints a summary including quote-quality and Gap figures.
2. `train` trains the Agent on the train period, scoring it on the validation period at each Checkpoint, saving the best model and writing a learning-curve CSV and PNG.
3. `evaluate` loads the best Checkpoint's model (the final model if no Checkpoint qualifies) and scores the Agent against both Baselines on the test period, Segment by Segment, reporting net return, Sharpe and max drawdown.

Vocabulary used throughout the repo is defined in [GLOSSARY.md](GLOSSARY.md).

## What the numbers mean

This is a scope-limited MVP, not a trading system. Read [docs/limitations.md](docs/limitations.md) before interpreting any result: it lists what the MVP deliberately does not model, explains the `gap_threshold` choice, and states the spec's acceptance bar.

## Tooling

Python 3.13, managed with [uv](https://docs.astral.sh/uv/). Run everything through `uv run` (for example `uv run ruff check`, `uv run ty check`).
