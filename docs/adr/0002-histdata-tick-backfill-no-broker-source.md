# Backfill from HistData ticks, no broker-tied source in ingestion

Historical backfill uses HistData.com tick bid/ask files, downloaded into the write-once raw cache. Ticks make `spread_mean` and `tick_count` tick-exact, so the Data contract stands unamended. Timestamps are New York local time with DST (verified by the [HistData tick probe](https://github.com/shadowgate15/shadow-trader/issues/22)), with one exception found by the real-data acceptance run (see Amendment): from the week opening 2019-03-10, weeks where US and EU daylight saving differ use fixed UTC-5. Each Session week's rule is inferred from its ticks and converted to UTC at ingestion. Ingestion names no live vendor: it is specified against the source-agnostic quote-stream interface, with a replay adapter feeding HistData ticks through the live path. The first concrete live adapter is chosen with the Broker integration map. Datasets built from HistData are private and never redistributed.

## Considered Options

- **OANDA v20 Practice (bid/ask candles from 2005)**: rejected for now. It needs a broker account, and the API licence (read: Asia Pacific text, Feb 2023; the US text may differ) limits copies of the Rates to installation and backup needs, requires a log of copies, and requires destroying all Licensed Materials on termination. That sits badly with a durable, immutable Dataset archive. It also only approximates `spread_mean` from M1 closes. Its terms permit "research and analysis" and "data processing" for Internal Use but say nothing on ML training.
- **Dukascopy**: rejected; terms say the site "may not be used to construct a database of any kind".
- **TrueFX**: not adopted; depth and terms are gated behind registration. May return as an optional cross-check.
- **Paid vendors (Tiingo, Twelve Data, Massive, Alpha Vantage)**: over the cost ceiling or lacking bid/ask, and free sources supply both sides.

## Consequences

- HistData publishes no terms (only "use at your own risk"), so Datasets built from it must stay private.
- Spreads and timestamps may differ from the eventual live broker; that gap is a Broker integration / Execution model concern, not an ingestion one.
- Files arrive as monthly zips, fetched by hand or a fragile script, which the write-once raw cache absorbs.
- Earliest tick month per pair was verified by the HistData tick probe: 2000-05 to 2000-06 for six majors, 2005-08 for NZD/USD; density is thin before ~2009.
- Session alignment (first tick of each week at the 17:00 New York open) is checked at build time as a warning, since the timestamp convention was verified by sampling, not exhaustively.

## Amendment: per-week raw-clock rule

The 2024-01 to 2024-04 acceptance run raised `session_alignment` warnings for March 2024. Checking the whole cache (2009-01 to 2026-09) against the 17:00 New York open/close, and against 08:30 New York news-release spikes (NFP, CPI), showed the raw clock is New York local in every week except 25 weeks from 2019-03-10 onward, all in the gaps between US and EU daylight-saving changes (mid to late March, late October to early November). Those weeks run on fixed UTC-5. The convention change cannot be dated more precisely than the first such gap week.

Decision: ingestion infers the rule per Session week (New York local or UTC-5) from the week's first and last ticks, records the offsets in the manifest and build report as provenance (not hashed; they are a function of the raw bytes already pinned), and fails the build for any week that fits neither rule or cannot be classified from itself or agreeing neighbours. A failing week is resolved by the user acknowledging it missing, not by guessing.

- Ambiguous or nonexistent local times can no longer fail the build; the offset is fixed per week and the DST change falls while the market is closed, so such a tick is dropped as out of session with the usual warning.
- A calendar rule backs the inference: before 2018-12-16 the stamps are `America/New_York` local; from 2018-12-16 they are Europe/Zurich local minus six hours (UTC-5 in EU winter, UTC-4 in EU summer). An independent project ([radiusred/tradedesk-marketdata](https://github.com/radiusred/tradedesk-marketdata), Apache-2.0) documents the same rule, cross-checked against Dukascopy. Over the whole cache, every week decided from its own ticks agrees with it. A week the ticks cannot decide takes the neighbours' rule if they agree, else the calendar rule (recorded as `calendar`, also when it overrules a week decided by one end only); it fails only if its ticks contradict every rule.
- On the EU fall-back Sunday of 2019, 2020, 2021, 2022, 2023 and 2025 the file lists the raw hour 19:00-19:59 twice, byte for byte. The repeat is dropped and recorded under `session_clock.replays`; any other backward step of 60 s or more fails the build. A tick under 60 s behind an already closed slot is dropped with the `late_ticks_dropped` warning.
- Weak spots to audit in the manifest: weeks marked `calendar`, `inherited` or `single_end` (mostly 2023, plus 2024-10-27 and 2026), and the 2023 weeks whose first tick is 60-120 minutes late.
