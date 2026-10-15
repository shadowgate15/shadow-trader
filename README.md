# Shadow Trader

A reinforcement learning agent designed to trade stocks. The agent uses deep learning techniques to learn trading strategies based on historical stock data.

## Market data ingestion

The `shadow-trader ingest` command builds immutable [Datasets](GLOSSARY.md) of [Bars](GLOSSARY.md) for forex [Instruments](GLOSSARY.md) from HistData tick files (see [ADR 0002](docs/adr/0002-histdata-tick-backfill-no-broker-source.md)) and runs the live ingester that appends [Revisions](docs/adr/0003-append-only-revisions-for-live-bars.md). Supported Instruments are EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD and NZDUSD. Bar sizes are the whole-minute divisors of 60: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30 and 60.

Run everything through `uv`:

```sh
uv run shadow-trader ingest <command> --config ingest.toml [flags]
```

### Typical workflow

1. `fetch` the raw months into the cache. This is the only command that touches the network.
2. `build` a backfill Dataset from the cache.
3. Optionally, `live` to run the live ingester, `build-live` to turn its sealed Bars into a live Dataset, and `diff` to compare that Dataset against a backfill Dataset.

### Commands

Every command requires `--config` (path to the TOML file) and accepts the three overrides below. Nothing else can be overridden from the command line.

| Flag | Overrides |
| --- | --- |
| `--instruments EURUSD,GBPUSD` | `build.instruments` (comma-separated) |
| `--start YYYY-MM`, `--end YYYY-MM` | `build.start`, `build.end` (UTC months, inclusive) |
| `--bar-sizes 5,15` | `build.bar_sizes` in minutes (comma-separated) |

| Command | What it does | Extra flags |
| --- | --- | --- |
| `fetch` | Downloads each missing month into the write-once raw cache. Cached months are skipped without a network call; months absent at the source (an empty download token) are reported as warnings. Failed attempts are retried with exponential backoff, each with a fresh token. A month is fetched only once it has settled, 7 days after it ends, so the write-once cache never freezes a still-changing file; months not yet settled are reported as warnings and picked up by a later run. `end` defaults to the last settled UTC month. | none |
| `build` | Builds one backfill Dataset per configured bar size from the raw cache and writes a build report. A [Gap](GLOSSARY.md) is flagged, never filled. | none |
| `live` | Runs the live ingester once per Instrument, writing the raw quote log and run reports. Runs a single bar size, so pass `--bar-sizes N` if the config lists several. The only adapter in v1 is `replay`, which feeds the cached HistData ticks through the live path with no network. | none |
| `build-live` | Builds a live Dataset per bar size from the raw quote log, over sealed slots only (after the [repair window](GLOSSARY.md)), using the latest Revision of each Bar. | `--from`, `--to`: the range `[from, to)`, ISO dates or datetimes (naive values are UTC). Both are required, and `--from` must be before `--to`. |
| `diff` | Writes an advisory reconciliation report comparing a live Dataset with a backfill Dataset. It is never used as repair, and the Datasets are never merged. Refuses Datasets that are not comparable. | `--live LIVE_ID`, `--backfill BACKFILL_ID`: the two `dataset_id`s (both required). |

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success. Warnings (for example months absent at the source) still exit 0. |
| 1 | Build failure: a rejected download, a failed `build`, `live` or `build-live` run, or a refused `diff`. |
| 2 | Config error: a missing or invalid config, an unknown key, a missing required flag value, or an unknown flag. |
| 130 | Interrupted with Ctrl-C. One `interrupted during <phase>` line on stderr, no traceback. |

### Progress and status

Long commands report that they are working. Progress is drawn with [rich](https://rich.readthedocs.io) and goes to **stderr**; **stdout** carries only the final result lines (`built ...`, `fetched ...`, `report: ...`), exactly as before, so scripts can keep parsing it.

| Flag | Effect |
| --- | --- |
| (default) | Progress on, tolerated warnings shown as they happen (the first 20; the rest are counted). |
| `-q`, `--quiet` | No progress and no warnings: only the final result lines and errors. |
| `-v`, `--verbose` | Per-month detail (including cached months in `fetch`) and every tolerated warning. |

`-q` and `-v` are exclusive and work on every command, before or after it (`ingest -q build ...` or `ingest build ... -q`).

- `fetch`: one outcome per month (`fetched`, `cached`, `absent`, `unsettled`, `rejected`) and a final tally.
- `build`: phases `scanning clocks`, then `building months` (n/total, current month), then `writing dataset`. Dropped replay hours, clock-rule fallbacks, late ticks, thin data and misaligned opens are printed as `warning: ...` when found; the end-of-run summary and build report are unchanged. A failure names the phase and the month, for example `build failed during scanning clocks at EURUSD/202401`.
- `build-live` and `diff` show their phases; `live` shows a status line (bar emissions, age of the last tick, Session state), refreshed on each closed Bar.
- On an interactive terminal each phase is a live bar with spinner, n/total, elapsed time, ETA and the current month. When stderr is not a terminal (pipes, CI) or `TERM=dumb`, the output is plain lines with no control characters: one per month (`[3/48] EURUSD 202403`), phase start and end lines with timings, and for `live` at most one status line a minute. `NO_COLOR` drops colour.

```text
scanning clocks: 2 step(s)
[1/2] EURUSD 202401
[2/2] EURUSD 202402
scanning clocks: done in 0.9s
building months: 2 step(s)
...
```

### Config

The config is a TOML file. Unknown keys are rejected. Only `[paths] data_root` is required; `build.instruments` must also be non-empty, either in the file or via `--instruments`.

```toml
[paths]
data_root = "/data/shadow-trader"

[build]
instruments = ["EURUSD"]
bar_sizes = [5]
start = "2024-01"
end = "2024-03"
```

| Table | Keys (defaults) |
| --- | --- |
| `[paths]` | `data_root` (required): root of everything the ingester writes. |
| `[build]` | `instruments` (none; at least one), `bar_sizes` (`[5]`), `start` (`"2009-01"`), `end` (last complete UTC month), `acknowledged_missing` (`[]`): a list of `{ instrument, month, reason }` tables for months known to be absent at the source. |
| `[instruments.<PAIR>]` | `price_scale` (5 decimal places, 3 for JPY pairs). |
| `[rules.gap]` | `outage_threshold_minutes` (30). |
| `[rules.anomaly]` | Toggles `crossed_quote`, `zero_spread`, `price_spike`, `stale_quote`, `spread_spike` (all `true`) and their thresholds: `price_spike_window_ticks` (50), `price_spike_k` (10.0), `price_spike_floor_pips` (10.0), `stale_quote_min_ticks` (20), `stale_quote_min_minutes` (5), `spread_spike_k` (10.0), `spread_spike_floor_pips` (2.0), `spread_spike_lookback_hours` (24), `alert_fraction` (0.01). |
| `[reconciliation]` | `diff` tolerances: `price_pips` (0.5), `spread_pips` (0.3), `spread_relative` (0.3), `tick_ratio_min` (0.5), `tick_ratio_max` (2.0). |
| `[live]` | `adapter` (`"replay"`), `grace_seconds` (1), `repair_window_minutes` (60). |

Content-affecting values feed the `dataset_id` hash, so changing one produces a new Dataset: each Instrument's `price_scale`, its `acknowledged_missing` months (not the free-text `reason`), and the `[rules.gap]` and `[rules.anomaly]` values, except `alert_fraction`. Reporting-only values never change a Dataset's identity: `alert_fraction`, all of `[reconciliation]`, and all of `[live]`. Only the Instruments being built count towards the hash, so editing another Instrument's settings does not change it.

### Where outputs land

All paths are under `data_root`:

| Path | Contents |
| --- | --- |
| `raw/histdata/<INSTRUMENT>/<YYYYMM>.zip` | The write-once raw cache, byte-exact, each file with a `.meta.json` sidecar. A corrected re-download is a new `.r2` file, never an overwrite. |
| `raw/live/<source>/<INSTRUMENT>/` | The live raw quote log, in hourly files. |
| `datasets/<dataset_id>/` | An immutable Dataset: Parquet partitions per Instrument and year, `manifest.json` (written last) and `build-report.json`. |
| `reports/failed/` | One JSON report per failed build. |
| `reports/live/<source>/<INSTRUMENT>/` | One report per live run. |
| `reports/reconciliation/` | `reconciliation-<live_id>-<backfill_id>.json` from `diff`. |

### Tests

`uv run pytest` skips the slow acceptance tests (`-m 'not slow'` is the default). To run them against real EUR/USD history, point them at a raw cache holding the months you want to cover (fetched with `ingest fetch`); each test is range-driven and uses only the months its range names, so a partial cache is fine:

```sh
SHADOW_TRADER_RAW_ROOT=/data/shadow-trader uv run pytest -m slow
```

`SHADOW_TRADER_RAW_ROOT` is the `data_root` holding the raw cache. `SHADOW_TRADER_REPLAY_RANGE` (`YYYY-MM` or `YYYY-MM:YYYY-MM`, default `2024-01`) picks the range for the replay-versus-batch parity test, and `SHADOW_TRADER_BUILD_RANGE` (same form, default every cached month) the range for the build and `dataset_id` reproducibility test. The raw-clock calendar cross-check reads every cached month and checks each week decided on tick evidence against the HistData calendar rule (New York local before 2018-12-16, Europe/Zurich minus 6 h after). The build drops an hour that HistData lists twice (a byte-identical block, seen on the EU clock-change Sunday in 2019-2023 and 2025) and records it under `replays` in the manifest's `session_clock`; any other backward step in the stamps fails the build. A tick up to a minute behind a slot that has already closed (seen in the 2026 files) is dropped and counted under the `late_ticks_dropped` warning.

## Dataset preparation

`shadow-trader prepare build --config prepare.toml` turns one named ingestion Dataset into train, validation and test prepared Datasets plus a train-only scaler; `shadow-trader prepare verify BUILD_ID --config prepare.toml` audits a stored build. Python API: `build_prepared`, `open_prepared`, `load_scaler` / `Scaler.apply`, `compute_features`, and for live use `FEATURE_SPEC`, `evaluate_tail`, `check_compat`, `check_tail_parity` (all in `shadow_trader.preparation`). Design: GLOSSARY.md and `docs/adr/0004-*`.

Exit codes for `prepare`: 0 success, 1 build failure (or a `verify` check failing), 2 config error or a malformed `BUILD_ID`, 130 interrupted with Ctrl-C (one `interrupted during <phase>` line on stderr, no traceback). A build is complete only once its build record exists. An interrupt during the `write` phase can leave the train-only scaler and some Datasets in the store with no build record. These orphans are harmless: nothing references them, an identical rebuild reuses them under the same ids, and an incomplete Dataset directory is discarded and rewritten. They are not cleaned up on interrupt.

Tests: `uv run pytest tests/preparation`. The opt-in real-data smoke (`tests/preparation/test_real_data_smoke.py`) runs with `SHADOW_TRADER_RAW_ROOT=/data/shadow-trader uv run pytest -m slow tests/preparation`; it needs a built 5-minute EURUSD ingestion Dataset spanning over a year under that root and is skipped otherwise.

Polars upgrades: the full preparation suite (goldens, parity, causality, leakage) must pass; any changed golden means a new feature spec version. See the upgrade rule in ADR 0004.
