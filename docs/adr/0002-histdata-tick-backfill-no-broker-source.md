# Backfill from HistData ticks, no broker-tied source in ingestion

Historical backfill uses HistData.com tick bid/ask files, downloaded into the write-once raw cache. Ticks make `spread_mean` and `tick_count` tick-exact, so the Data contract stands unamended. Timestamps are EST without DST (fixed UTC-5) and are converted to UTC at ingestion. Ingestion names no live vendor: it is specified against the source-agnostic quote-stream interface, with a replay adapter feeding HistData ticks through the live path. The first concrete live adapter is chosen with the Broker integration map. Datasets built from HistData are private and never redistributed.

## Considered Options

- **OANDA v20 Practice (bid/ask candles from 2005)**: rejected for now. It needs a broker account, and the API licence (read: Asia Pacific text, Feb 2023; the US text may differ) limits copies of the Rates to installation and backup needs, requires a log of copies, and requires destroying all Licensed Materials on termination. That sits badly with a durable, immutable Dataset archive. It also only approximates `spread_mean` from M1 closes. Its terms permit "research and analysis" and "data processing" for Internal Use but say nothing on ML training.
- **Dukascopy**: rejected; terms say the site "may not be used to construct a database of any kind".
- **TrueFX**: not adopted; depth and terms are gated behind registration. May return as an optional cross-check.
- **Paid vendors (Tiingo, Twelve Data, Massive, Alpha Vantage)**: over the cost ceiling or lacking bid/ask, and free sources supply both sides.

## Consequences

- HistData publishes no terms (only "use at your own risk"), so Datasets built from it must stay private.
- Spreads and timestamps may differ from the eventual live broker; that gap is a Broker integration / Execution model concern, not an ingestion one.
- Files arrive as monthly zips, fetched by hand or a fragile script, which the write-once raw cache absorbs.
- Earliest tick month per pair is unverified until a probe downloads real files.
