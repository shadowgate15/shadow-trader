# Data source research (issue #14)

Question: which source(s) can feed Market data ingestion? See issue #14 and map #13. Researched 2026-10-12 (all "accessed" dates below are 2026-10-12). Throwaway research branch, never merged.

**Status of this document:** findings plus a recommendation. The decision is the human's, in a later session.

## Requirements recap (from GLOSSARY.md, #2, #3)

The Bar contract needs, per Bar: bid-side OHLC, `spread_close`, `spread_mean`, optional `tick_count`, UTC clock-aligned half-open intervals, Gaps flagged (never filled). So a source must give **bid and ask** (or bid + spread), ideally at tick/quote level so `spread_mean` is exact. Pre-built bars can supply `spread_close` (ask close minus bid close) but only approximate `spread_mean`. Targets: EUR/USD >= 5 years, other USD majors, $0 recurring preferred, <= ~$25/month acceptable, personal ML-training licence, usable for live/paper, signup lead time reported.

Evidence labels: **[V]** verified on a vendor primary page this session; **[S]** vendor-adjacent or secondary source; **[U]** unverified (page was gated, JS-rendered or silent). Nothing marked [U] should be relied on without a spot check.

## Comparison

| Source | Tick/quote vs bars | Bid + ask | EUR/USD depth | Granularity | Cost | Personal ML-training licence | Live/paper | Mechanics and limits | Signup lead time |
|---|---|---|---|---|---|---|---|---|---|
| **OANDA v20 REST (fxTrade Practice)** | Pre-built bar candles (plus quote polling/stream) | Yes: `price=BA` returns separate bid and ask candles [V] | "dating back to 2005" [V]; history "Complete" [V] | S5, S10, S15, S30, M1 ... D, W, M [V] | $0 (demo account free) [V] | **Unclear**: token issued after accepting "the API license" [V]; licence text not read, ML-training clause unverified | Yes: same API has pricing poll and stream (at most 4 prices/s per instrument) [V]; also a broker for paper | 5000 candles/page, `from`/`to`, 120 req/s, 2 new connections/s, per IP [V] | Demo account + token, minutes to a day [U]; not offered in all divisions [V] |
| **HistData.com** | Tick and M1 files | Tick file has Bid and Ask [V]; M1 bars are bid-only [V] | Year selector lists 2000-2026 for EUR/USD tick [V], but the page template renders for any month so real start is [U] | Tick, M1 | $0 (paid FTP and Drive-sync extras) [V] | Terms page not found; FAQ says "no warranty... at your own will and risk" [V]; no explicit ML or commercial clause found | Historical only; updated monthly (page showed update 2026-10-11) [V] | One zip per pair/month by web click; no documented API or rate limit; timestamps are EST **without DST** (UTC-5 fixed) [V] | None (no account) |
| **Dukascopy historical feed** | Tick (bi5 binary) and export tool CSV tick to monthly [V] | Yes: records hold ask, bid and volumes [S] | EUR/USD tick from 2003-08-04 [S] | Tick up to monthly | $0 (export tool free to demo and live account holders [V]) | **Risky**: terms allow personal non-commercial use but say the site "may not be used to construct a database of any kind" [V] | Historical only (live needs a Dukascopy account and JForex) | `datafeed.dukascopy.com/datafeed/<PAIR>/<Y>/<month 0-11>/<D>/<H>h_ticks.bi5` [S, not vendor-documented]. Observed today: HTTP 503 then 429 from unauthenticated curl, so expect throttling | Export tool needs demo account [U]; bi5 URLs need none |
| **TrueFX** | Tick | "Top-of-book, tick-by-tick... fractional pip spreads", bid/ask quotes [V] | Pairs and start year are behind login [U] | Tick | $0, free registration required [V] | FAQ: indicative data usable for "historic back-testing"; full terms not read [V partial] | FIX streaming advertised on paid plans [V]; free tier is historical download | Monthly files behind login; limits [U] | Free registration, lead time [U] |
| **Interactive Brokers (TWS API)** | Pre-built bars via `reqHistoricalData` (BID_ASK etc.) | Yes: BID_ASK counts as two requests [V] | [U] | 1 min and larger limits "lifted" [V] | Account needed; data fees for forex [U] | Personal use under IB data terms [U] | Yes (paper account is the natural broker feed) | 60 requests/10 min, 6/2 s per contract, 50 open; BID_ASK double-counted [V] | Account approval, days to weeks [U] |
| **Alpha Vantage** | Pre-built bars | **No**: `FX_INTRADAY` returns only timestamp, open, high, low, close [V] | `full` output, premium only [V] | 1/5/15/30/60 min | Premium from $49.99/month [V]; free is 25 req/day, compact only [V] | Premium for "personal use" [V] | Realtime on premium | Over budget and no bid/ask | Instant |
| **Massive (formerly Polygon) Currencies** | Bars built from quoted bid/ask [V] | Bar fields not confirmed to carry both sides [U] | Basic: 2 years [V]; Starter: since 2009-09-25 [V] | Custom multiples | Basic $0 (EOD, 5 calls/min); Starter $49/month [V] | "Non-pros only" [V] | Starter real-time and WebSocket | 50,000 base aggregates per query [V] | Instant |
| **Twelve Data** | Pre-built bars | Bid/ask not confirmed [U] | Intraday depth on free plan [U] | [U] | Basic $0 (8 credits/min, 800/day) [V]; Grow $29/month [V] | [U] | Free plan lists real-time forex [V] | Credit-limited | Instant |
| **Tiingo FX** | Top-of-book quotes and resampled bars (`resampleFreq=5min`) [V] | Top-of-book has `bidPrice` and `askPrice` [V]; bars' side unconfirmed | [U] | 1 min to 1 day [S] | Individual Power tier $30/month [V]; whether FX is in the free tier [U] | Personal use only on free and Power [V] | Real-time top-of-book [V] | 50 req/hr and 1,000/day free; 10,000/hr and 100,000/day Power [V] | Instant |
| **FXCM API** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | API reported deprecated [S]; excluded | n/a |

### Coverage of the other USD majors

None of the vendor pages read this session enumerated all of GBP/USD, USD/JPY, USD/CHF, USD/CAD, AUD/USD, NZD/USD against the exact access path. HistData's FAQ lists "EUR/USD, EUR/CHF, EUR/GBP..." with many more [V, truncated]; TrueFX advertises "16+ major currency pairs" [V]; Dukascopy and OANDA are widely documented as carrying all majors but this was **not verified** here. Spot-check each pair's earliest month before adopting any source.

### How each fits the Bar schema

- **Tick sources (HistData, Dukascopy, TrueFX):** one Bar builder over bid/ask ticks yields exact `open/high/low/close` (bid), `spread_close`, `spread_mean`, `tick_count`, and Gaps = no ticks in an open Session interval. This is the best fit for the contract and the "one Bar builder shared with backfill" live design.
- **OANDA candles:** M1 bid and ask candles aggregate into 5 or 15 minute bars. Bid OHLC is exact; `spread_close` = ask close - bid close is exact at the M1 boundary; `spread_mean` can only be approximated (for example mean of per-M1 closes), and OANDA candle `volume` would need confirming as the `tick_count` proxy [U]. This is a **possible conflict with #3** worth a deliberate human call.
- **Timestamps:** HistData is fixed UTC-5; convert to UTC (add 5h) and do not apply DST. OANDA returns UTC; its alignment args only affect D/W/M [V].
- **OANDA history caveat:** the candles endpoint serves "base-price group" data, which can differ from the account's live pricing group [V], so backfill and paper-feed prices may not match exactly. Relevant to the shared-schema parity goal.

## Ranking (free first)

1. **OANDA v20 Practice** - $0, API-native, bid and ask candles back to 2005, usable for both backfill and live quote polling or stream. Weak on exact `spread_mean`; licence wording must be read.
2. **HistData.com** - $0, no signup, tick bid/ask so exact spreads; manual monthly downloads, no SLA or warranty, no live, unclear terms, EST-no-DST timestamps.
3. **Dukascopy** - $0, deepest well-known tick history (2003), but the "no database" clause is the largest licensing risk of any candidate and the bulk endpoint is unofficial and throttles.
4. **TrueFX** - $0 with free registration, tick bid/ask, but depth and pairs unverifiable without an account.
5. **Interactive Brokers** - data cost not guaranteed $0, needs an approved account; most relevant as a broker (paper feed), which is out of scope for this map.
6. **Paid, over ceiling or missing bid/ask:** Tiingo Power ($30), Twelve Data Grow ($29), Massive Starter ($49), Alpha Vantage ($49.99, no bid/ask). None is the "only way" to get bid and ask, since free sources provide it, so the paid exception is not triggered.

## Recommendation (a recommendation, not a decision)

- **Suggested primary: OANDA v20 Practice** for both backfill and the live quote-stream adapter. It is free, scriptable (5000 candles/page at 120 req/s means about a decade of M1 BA candles in a few thousand calls), has S5/M1 granularity to build 5 and 15 minute bars, and the same account type can later serve paper trading. Conditions: read the OANDA API licence for personal ML-training use first, and either accept an approximate `spread_mean` or amend the Bar contract to derive it from M1 closes.
- **Suggested fallback / cross-check: HistData tick files** if an exact `spread_mean` and `tick_count` from raw ticks is required. Dukascopy's ToS is a reason not to make it primary.
- If the human wants exact tick-derived spreads, the trade-off is between HistData (unclear terms, manual download) and accepting OANDA's approximation.

## Open questions for the human / before closing #14

1. Is an approximate `spread_mean` (from M1 bid/ask candle closes) acceptable, or must it be tick-exact?
2. Read and record the OANDA API licence and HistData / TrueFX terms regarding ML training before any choice is final.
3. Spot-check earliest months and file coverage for the six other USD majors on the shortlisted source.
4. Confirm OANDA demo-account eligibility from the user's jurisdiction (division restrictions apply) and the signup time.
5. Paper-feed source remains an external dependency (Broker integration map).

## Sources (accessed 2026-10-12)

- OANDA REST v20 introduction (token, API licence, "historical pricing back to 2005"): https://developer.oanda.com/rest-live-v20/introduction/
- OANDA instrument candles endpoint (listed within the Pricing page; price, granularity, count max 5000): https://developer.oanda.com/rest-live-v20/pricing-ep/
- OANDA candlestick granularities and bid/ask candle data: https://developer.oanda.com/rest-live-v20/instrument-df/
- OANDA rate limits, practice and live hosts: https://developer.oanda.com/rest-live-v20/development-guide/
- OANDA API comparison (history "Complete", S5 to M): https://developer.oanda.com/rest-live-v20/api-comparison/
- OANDA troubleshooting (historical vs live pricing group): https://help.oanda.com/us/en/faqs/rest-v20-api-troubleshooting-guide.htm
- HistData FAQ: https://www.histdata.com/f-a-q/
- HistData EUR/USD tick download pages: https://www.histdata.com/download-free-forex-data/?/ascii/tick-data-quotes/eurusd
- Dukascopy historical data export: https://www.dukascopy.com/swiss/english/marketwatch/historical/
- Dukascopy terms of use: https://www.dukascopy.com/swiss/english/legal-pages/terms-of-use/
- Dukascopy tick start dates (secondary): https://tickstory.com/dukascopy-historical-data-available-date-ranges ; bi5 URL layout and record fields (secondary, search summary): https://github.com/theorycraft-trading/dukascopy
- TrueFX: https://www.truefx.com/truefx-historical-downloads/ , https://www.truefx.com/truefx-market-data-faq/ , https://www.truefx.com/
- IBKR historical data limitations: https://interactivebrokers.github.io/tws-api/historical_limitations.html
- Alpha Vantage docs and premium pricing: https://www.alphavantage.co/documentation/ , https://www.alphavantage.co/premium/
- Massive pricing and forex aggregates: https://massive.com/pricing?product=currencies , https://massive.com/docs/rest/forex/aggregates/custom-bars
- Twelve Data pricing: https://twelvedata.com/pricing
- Tiingo forex docs and pricing: https://www.tiingo.com/documentation/forex , https://www.tiingo.com/about/pricing
- FXCM API deprecation (secondary): https://www.esegece.com/community/blog/fxcm-api-removed-from-sgcwebsockets/
