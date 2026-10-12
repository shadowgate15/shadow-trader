# Market data ingestion: implementation spec

Status: **draft**. All decisions below are settled on the wayfinder map [Map: Market data ingestion spec](https://github.com/shadowgate15/shadow-trader/issues/13). Section 12 lists the questions still open; the spec is implementation-ready only once those tickets are closed and folded in.

Vocabulary follows `GLOSSARY.md`. Upstream contracts ([Market conventions](https://github.com/shadowgate15/shadow-trader/issues/2), [Data contract](https://github.com/shadowgate15/shadow-trader/issues/3)) are inputs, not reopened here.

## 1. Problem and scope

Market data ingestion turns raw quotes into immutable, versioned Datasets of Bars, and into a live stream of closed Bars with the same schema.

In scope: source choice, Bar building, Gap vs outage semantics, anomaly flagging, Dataset layout and versioning, live-mode semantics.

Out of scope: choosing the paper-feed source (Broker integration map); cleaning, cross-instrument alignment, features, splits, scaling, fill modelling, holiday calendar; tick or order-book level trading.

Constraints: Python 3.13 / uv, local Mac, free data in v1. First Instrument EUR/USD. Default bar size 5 min, fallback 15 min.

## 2. Source (ADR 0002)

- **Backfill**: HistData.com tick bid/ask files, fetched into the raw cache. Tick-exact `spread_mean` and `tick_count`; no Data contract change.
- **Fetch mechanics**: no account. `POST https://www.histdata.com/get.php` with `tk` (token from the month page's hidden form), `date=<year>`, `datemonth=<YYYYMM>`, `platform=ASCII`, `timeframe=T`, `fxpair=<PAIR>`, `Referer` set to the month page. A 0-byte HTTP 200 means either "no such month" or a transient failure: retry before concluding absence, and validate the zip. Year pages list all 12 months regardless of coverage, so only a download proves coverage. No checksums are published; no rate limit was hit at ~400 requests spaced 0.2-0.5 s. The scrape is undocumented and fragile; the write-once raw cache absorbs that.
- **File layout**: zip `HISTDATA_COM_ASCII_<PAIR>_T_<YYYYMM>.zip` holding `DAT_ASCII_<PAIR>_T_<YYYYMM>.csv` and a `.txt` status report. CSV has no header: `YYYYMMDD HHMMSSmmm,bid,ask,0`, LF line ends, millisecond resolution, 6-decimal price strings, last column always 0 and ignored.
- **Coverage (verified)**: first tick month EUR/USD, GBP/USD, USD/JPY, USD/CHF 2000-05; USD/CAD, AUD/USD 2000-06; NZD/USD 2005-08. EUR/USD is contiguous to the current month (current month is partial). Data is thin before about 2009 (EUR/USD January rows: ~71k in 2001, ~401k in 2009, ~2.2M in 2024). Real holes exist (EUR/USD Friday 2005-07-01 absent), so the Gap rules must tolerate them.
- **Timestamps**: naive New York local time **with DST**, converted to UTC via `America/New_York` (`zoneinfo`). The market is closed at both DST edges, so a clean feed has no ambiguous or nonexistent local times; any such tick is a hard build error naming file and row, never a silent fold choice.
- **Session-alignment check**: the first tick of each week should fall at the 17:00 New York open; weeks off by more than 1 hour produce a warning (not a failure). Exact tolerance is set under verification (section 12).
- **Live**: no vendor chosen. A replay adapter feeds HistData ticks through the live path and is the only v1 adapter. The first real live adapter is chosen with the Broker integration map and is an external dependency here.
- **Rejected**: OANDA v20 (needs broker account, approximates `spread_mean`, licence terms), Dukascopy (no-database clause), TrueFX (gated; optional later cross-check), paid vendors.
- **Accepted risk**: HistData publishes no terms, so HistData-derived Datasets stay private and are never redistributed. Spread and timestamp differences versus the eventual live broker belong to Broker integration / Execution model.

## 3. Output: Bar schema (Data contract, unchanged)

One series per Instrument and bar size, persisted as Parquet with an explicit schema:

| Column | Notes |
|---|---|
| `instrument` | e.g. EUR/USD |
| `open_time` | UTC, half-open `[open, open + size)`, identifies the Bar |
| `open`, `high`, `low`, `close` | bid side; ask = bid + spread |
| `spread_close`, `spread_mean` | price units, not pips |
| `tick_count` | nullable int; activity proxy only, never liquidity |
| `is_gap` | true on Gap rows; prices, spreads, `tick_count` null |
| `session_id` | Session open time, UTC; on every row including Gaps |
| `is_rollover` | true on the Bar opening exactly 17:00 New York Mon-Thu |
| `quality_flags` | list of strings, empty = clean |

No `volume`. No rows for the weekly closure. No adjustments at ingestion.

## 4. Bar building (ticket: Bar-building rules)

1. Each configured bar size is built directly from ticks in one pass over the raw cache. No intermediate 1-minute layer; one Dataset per bar size.
2. Bar sizes are whole-minute divisors of 60 (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60). Anything else is a config error. Default 5.
3. Per Bar, ticks ordered by timestamp (file order breaks ties): `open` = first tick bid, `close` = last tick bid, `high`/`low` = max/min bid, `spread_close` = last tick `ask - bid`, `spread_mean` = arithmetic tick mean of `ask - bid` (not time-weighted), `tick_count` = tick count. `open` is never the previous close.
4. Determinism: prices and spreads are scaled integers (5 dp, 3 dp for JPY pairs) for min/max/sum/count, converted once to float64 for storage; `spread_mean` = integer sum / count.
5. A slot inside a Session with zero ticks is a Gap row. Never carried forward.
6. Quotes-only: the builder accepts quotes only (HistData, replay, live adapters). Coarser-only sources are unsupported in v1; adopting one needs its own decision and likely a Data contract amendment.
7. Session grid: dense from the weekly open (Sun 17:00 New York) up to, excluding, the weekly close (Fri 17:00 New York), computed per Session with `America/New_York`. The US clock change happens Sun 02:00 New York while closed, so each Session has one UTC offset (open 21:00 or 22:00 UTC) and is exactly 120 h with a constant slot count per bar size. No in-Session DST edge.
8. Ticks outside a Session are dropped from Bars, counted in the build report and manifest, and raise a warning (not a failure).
9. No cleaning in the builder: every parsed tick enters OHLC and spread statistics, spikes and crossed quotes included.

## 5. Gap vs outage (ticket: Gap versus source outage)

- `is_gap` stays the only "no data" bit. The outage distinction is a `quality_flags` value on Gap rows only; Gap rows stay null-priced. No schema change.
- Meanings: plain Gap = no quotes arrived, cause unknown; `suspected_outage` = absence likely a feed problem; `source_missing` = known absent from the source. None is ever filled. Handling (mask, skip, penalise) belongs to dataset preparation and the environment.
- `suspected_outage` (backfill): a run of consecutive Gap Bars inside one Session whose duration is at least a threshold (default 30 min, configurable, a duration not a Bar count). No cross-instrument corroboration. The rule and threshold are recorded under the manifest's **Gap rule-set version**.
- Missing raw file or period (absent, empty, corrupt): the build fails by default, naming the period. An explicit acknowledged-missing list in config turns those periods into Gap rows flagged `source_missing`; the list is recorded in the manifest.
- Coverage edges: the dense grid exists only inside the declared time range intersected with Sessions. Anything inside it lacking source coverage (e.g. before an Instrument's first tick) follows the missing-period rule.

## 6. Anomaly flags (ticket: Anomaly rules for quality_flags)

A detector pass separate from the builder, over the same ticks, in two tiers. A Bar is flagged if any of its ticks trips a tick rule or the Bar trips a Bar rule. Values are never altered or dropped.

- **Vocabulary** (closed, lowercase snake_case, additive-only): `crossed_quote`, `zero_spread`, `price_spike`, `spread_spike`, `stale_quote`, plus `suspected_outage` and `source_missing` (Gap rows only). `negative_spread` is dropped as redundant with `crossed_quote`. Names are never repurposed; retired names stay reserved. Additions bump the anomaly rule-set version (new Dataset hash), not `schema_version`.
- **Causality**: every rule is look-back-only; rolling state carries across Sessions. Baseline-dependent flags are suppressed until the baseline is warm; baseline-free rules always apply. The same detector runs live, so backfill and live flag identically.
- **Config**: one rule-set block with Instrument-class defaults (pip 0.0001; JPY 0.01), recorded resolved in the manifest as the **anomaly rule-set version**. Any change gives a new Dataset hash. Rules can be disabled; the vocabulary is fixed.

| Flag | Tier | Rule | Defaults |
|---|---|---|---|
| `crossed_quote` | tick | ask < bid | none |
| `zero_spread` | tick | ask == bid | none |
| `price_spike` | tick | bid further than `max(k × robust_scale, floor)` from the median bid of the previous N ticks; `robust_scale` = scaled MAD of tick-to-tick bid changes | N=50, k=10, floor=10 pips |
| `stale_quote` | tick | run of identical bid/ask ticks reaching >= 20 ticks and >= 5 min; flagged on every Bar containing a tick of the run from then until the quote changes | 20 ticks, 5 min |
| `spread_spike` | Bar | `spread_mean` > `max(k × median, median + floor)`, median over non-Gap Bars in the previous 24 h (duration, so bar-size independent); rollover Bars not exempt | k=10, floor=2 pips |

Silence with no ticks is a Gap / `suspected_outage`, so `stale_quote` never overlaps it. Reporting: manifest and build report record per-flag counts (Bars and ticks); a warning (not a failure) fires if more than 1% (configurable) of non-Gap Bars carry `price_spike`, `crossed_quote` or `stale_quote`. `zero_spread` and `spread_spike` are excluded from the warning.

## 7. Dataset layout and versioning (ADR 0001)

- Layout: `<data_root>/datasets/<dataset_id>/manifest.json` and `bars/instrument=<X>/year=<YYYY>/part-0.parquet`. Bar size is a Dataset attribute, not a partition. `data_root` is configurable and outside the repo. Each file holds the dense Session grid, Gap rows included.
- Versioning: append or fix writes a new Dataset directory. `dataset_id` is the content hash; the manifest records `parent`. Unchanged closed partitions are hardlinked from the parent; only the open year is rewritten. A `latest` pointer per series is allowed but never counts as lineage; Candidates name the full hash.
- Content hash: Merkle over logical content. Per partition, sha256 of rows in canonical order (sorted by `open_time`, fixed column order, canonical Arrow encoding, explicit nulls). Dataset hash = sha256 over `schema_version`, bar size, and sorted (partition key, partition hash). Provenance (created_at, fetch times, builder revision) is not hashed, so identical raw inputs and rules reproduce the same hash.
- Manifest (JSON, written last by atomic rename; no manifest = incomplete, never read): `source`, bar size, Instruments, time range, row count, content hash, `dataset_id`, `parent`, `schema_version`, per-partition entries (path, rows, gap count, hash), `raw_inputs` (raw cache file hashes), `builder_version` and git revision, anomaly and Gap rule-set versions, `created_at`, per-Instrument Session coverage summary (Sessions, Gaps, flagged anomalies).
- `schema_version`: single integer, bumped on any change to columns, types or meaning; stored in the manifest and Parquet metadata. Readers refuse unknown versions. No migration: re-ingest from the raw cache into a new Dataset.
- Raw cache: ingest reads only from it, never the network. Write-once, byte-exact: `<data_root>/raw/<source>/<instrument>/<period>.<ext>` with a `.meta.json` sidecar (sha256, request params, fetch time, HTTP details). Never overwritten; a corrected re-download is a new file whose sidecar notes the supersession. No automatic pruning. For HistData: `<period>` is `YYYYMM`, `<ext>` is `zip`.

## 8. Live mode (ticket: Live mode semantics, ADR 0003)

1. **Close rule**: by event time. Adapters emit quotes and heartbeats with source UTC timestamps; the watermark is the latest source timestamp; a Bar closes at slot end plus grace (default 1 s). Empty slots close as Gap Bars once a heartbeat passes. No slots exist in the weekly closure.
2. **Revisions**: emitted Bars carry `revision` (from 0). A late or catch-up quote inside the repair window (default 1 h) re-emits the slot as the next revision; earlier revisions are kept. After the window the Bar is sealed and later quotes are dropped, counted and warned. As-seen view = revision current at a moment (decision audit); latest view = highest revision (running context window, Datasets).
3. **Reconnects**: adapters report source-down intervals; Gap Bars overlapping one are flagged `suspected_outage` at emission regardless of the run-length rule. Adapters may offer `fetch_quotes(start, end)` for catch-up, fed through the same builder as late arrivals; without it there is no repair. Priced Bars overlapping an outage are emitted unflagged. Intervals go in the live run report and Dataset manifest.
4. **Raw capture**: every live quote is appended (source and arrival time) to a write-once log under `<data_root>/raw/live/<source>/<instrument>/<period>.<ext>` with a `.meta.json` sidecar. Live Datasets are built only from this log.
5. **One builder**: a single incremental core (push quote or heartbeat, pull closed Bars). Backfill is the same core fed from raw files with a final flush.
6. **Live to Dataset**: an explicit build command over a closed range, sealed slots only, latest revisions. Manifest: `source` = live adapter id, `raw_inputs` = log files, run and source-down intervals, `parent` = previous live Dataset with hardlinked closed partitions. The run-length rule is re-applied across the series and unioned with live flags (live flags never recomputed away).
7. **Live vs backfill**: separate Datasets with different `source`, no merge, no parent edge. Backfill is canonical for training; a live Dataset is the record of what a paper or live run saw. An advisory reconciliation diff (Bars beyond tolerance, `tick_count`, spread drift) is the only link and never blocks or rewrites.
8. **Quote-stream interface**: events `Quote(ts, bid, ask)`, `Heartbeat(ts)`, `SourceDown(start)`, `SourceUp(ts)`, plus optional `fetch_quotes`. Timestamps are non-decreasing except catch-up output; the builder orders out-of-order quotes within an open slot by timestamp, then arrival.
9. **Starts, stops, crashes**: a mid-Session start emits from the first slot beginning at or after the start; earlier slots are not Gaps. Ingester downtime between runs counts as a source-down interval (Gap rows `suspected_outage` at Dataset build, intervals in the manifest). A clean shutdown flushes only genuinely closed slots.
10. Out-of-Session quotes are dropped and counted, as in section 4.

## 9. Build failures vs warnings

| Condition | Result |
|---|---|
| Missing/empty/corrupt raw period not acknowledged | fail |
| Ambiguous or nonexistent local time (DST edge tick) | fail, naming file and row |
| Unsupported bar size | fail (config error) |
| Ticks outside a Session | warning, counted |
| Session-alignment off by more than 1 h | warning |
| > 1% flagged Bars (`price_spike`, `crossed_quote`, `stale_quote`) | warning |
| Live quote after seal | dropped, counted, warning |

## 10. Decisions with ADRs

- `docs/adr/0001-content-hashed-immutable-datasets.md`
- `docs/adr/0002-histdata-tick-backfill-no-broker-source.md`
- `docs/adr/0003-append-only-revisions-for-live-bars.md`

## 11. External dependencies

- Live paper-feed source: Broker integration map. Ingestion consumes it only through the quote-stream interface.
- Consumers of Datasets (dataset preparation, environment) own Gap and flag handling, view selection (as-seen vs latest) and cross-instrument alignment.

## 12. Open questions (not yet decided; tickets on the map)

These were in the map's "Not yet specified" fog and have no decision. They are not decided here.

- **Interface and configuration shape**: CLI/API for backfill, live and the live-Dataset build; config schema (grace, repair window, outage threshold, anomaly block, acknowledged-missing list); whether live Dataset rows carry `revision`; package and repo layout.
- **Verification strategy**: golden-data tests (one winter and one summer month, per the DST decision), reproducibility checks for re-ingest, stream-equals-batch property test, Session-alignment tolerance.
- **Run reporting**: format of the build report, live run report, and the live-vs-backfill reconciliation diff report.
- **Instrument coverage**: which pairs beyond EUR/USD are in scope for v1.
