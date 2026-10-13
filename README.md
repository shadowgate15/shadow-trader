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
| `fetch` | Downloads each missing month into the write-once raw cache. Cached months are skipped without a network call; months absent at the source are reported as warnings. `end` defaults to the last complete UTC month. | none |
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

`SHADOW_TRADER_RAW_ROOT` is the `data_root` holding the raw cache. `SHADOW_TRADER_REPLAY_RANGE` (`YYYY-MM` or `YYYY-MM:YYYY-MM`, default `2024-01`) picks the range for the replay-versus-batch parity test, and `SHADOW_TRADER_BUILD_RANGE` (same form, default every cached month) the range for the build and `dataset_id` reproducibility test. The raw-clock calendar cross-check reads every cached month and checks each week decided on tick evidence against the HistData calendar rule (New York local before 2018-12-16, Europe/Zurich minus 6 h after).
