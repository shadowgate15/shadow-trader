# Research: free EUR/USD M5 history sources

Ticket: shadowgate15/shadow-trader#129 (map: #128). Researched 2026-10-20. Findings only; no source is chosen here.

Evidence labels: **[verified]** read or fetched from the owning source during this research (probes were run with `curl` against the live hosts); **[secondary]** a statement from a source that does not own the fact; **[unverified]** could not be confirmed.

Scope note: the repository already contains `docs/adr/0002-histdata-tick-backfill-no-broker-source.md`, which rejected Dukascopy and OANDA and chose HistData ticks for the backfill. This ticket answers the broader question for the MVP map, so every candidate is re-checked here rather than taken from the ADR. Where the ADR is the only source for a claim it is marked **[secondary: ADR 0002]**.

## Summary table

| Source | M5 delivered as | Bid and ask for M5 | Depth (EURUSD) | Free access | Terms for programmatic caching |
|---|---|---|---|---|---|
| FXCM candle repository | No M5. M1 weekly files (plus H1, D1); M5 must be built from M1 | Yes: BidOHLC and AskOHLC in the same row | Files found for 2012 week 1 to 2026 week 41, with gaps listed below | No account, plain HTTPS | "Personal use", "abides by our EULA"; EULA text not retrieved |
| Dukascopy | Web export tool offers bars; undocumented `datafeed` host serves daily M1 bid and ask files | Yes: separate BID and ASK files | M1 file found for 2003-05-02 | No account, but rate-limited | Website terms forbid building a database and forbid bots |
| HistData | No M5. M1 bars (bid only) and ticks (bid and ask) | Ticks only; bars are bid only | M1 listed 2000 to 2026-10 | No account | No terms published beyond "own risk" (per ADR 0002) |
| TrueFX | Ticks only | Yes (top-of-book ticks) | Not stated on pages read | Free registration required | Not retrieved |
| OANDA v20 | Native M5 via API | Yes via API price parameter, not re-verified here | From 2005 per ADR 0002 | Account required | Licence limits copies, per ADR 0002 |
| Massive (ex-Polygon) free tier | Aggregates, any multiplier | Aggregates built from quotes; single series returned | 2 years on free plan | Free API key, 5 calls per minute | Individual use only |
| Alpha Vantage `FX_INTRADAY` | Native 5min | Single price | Not stated | Premium endpoint; free key limited to 25 requests per day | Not retrieved |
| Yahoo Finance via yfinance | Native 5m | Single price | 60 days for 5m | No account | "Personal use only" |
| Twelve Data free plan | Native 5min (not verified for forex on free plan) | Single price | Not stated | 8 credits per minute, 800 per day | Not retrieved |
| Tiingo FX | Native resample to 5min | Top-of-book bid and ask fields documented | Not stated | Forex not listed on free plan | Internal use only |
| Kaggle datasets | Depends on dataset | Mostly single price | Varies | Free download, Kaggle account needed to download files | Per dataset licence, provenance mostly unstated |

## 1. FXCM free candle repository

Primary sources: README of `github.com/fxcm/MarketData` (the owner's repository; repo not archived, last push 2026-07-09 per the GitHub API), and direct requests to `https://candledata.fxcorporate.com/`.

**Access.** URL pattern `https://candledata.fxcorporate.com/{periodicity}/{instrument}/{year}/{week}.csv.gz`, periodicities `m1`, `H1`, `D1` only. **[verified]** README text. A request for `m5/EURUSD/2020/10.csv.gz` and for `m15` returned HTTP 404. **[verified]** No account, no key, plain HTTPS. **[verified]** Rate limits: none stated in the README; none encountered in about 800 requests at 8 in parallel. **[verified]**

**M5 is not published.** Only M1, H1, D1 exist. M5 would have to be aggregated from M1. **[verified]**

**Bid and ask.** One file, one row per bar, columns `DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose`. Checked on the files for 2012 week 20 and 2020 week 10 (M1) and the 2020 D1 file. **[verified]** So bid OHLC and ask OHLC are both present natively per bar; no volume column. **[verified]** Spread per bar is therefore directly available (ask minus bid), at the resolution of each bar's close, not tick-exact.

**Coverage and currency.** The README says years 2017 to 2020. **[verified]** That is out of date: a sweep of `m1/EURUSD/{2012..2026}/{1..53}` returned HTTP 200 for 753 of 795 URLs. Files were found from 2012 week 1; my sweep did not go earlier than 2012, but spot requests for 2011 weeks 1, 20, 40 and 2010 week 10 returned 404, so 2012 looks like the start. **[verified]** The latest file found was 2026 week 41 (today is 2026-10-20; weeks 42 and later returned 404). Weekly files appear to be added after each week closes, but the update schedule is not stated. **[verified for the observed state, unverified for the schedule]**

**Missing weeks found by the sweep (404), EURUSD M1:** 2015/53, 2016/53, 2017/53, 2021/53, 2022/53 (probably weeks that do not exist), and weeks that likely should exist: 2019/1, 2020/1, 2024/1, 2025/1, 2025/2, 2024/35, 2024/51, 2024/52, 2025/29, 2025/30, and a continuous block 2026/18 through 2026/31 (14 weeks). Whether week 1 gaps are an indexing convention (partial year-boundary weeks filed under another year) is not documented and was not investigated. **[verified list, unverified cause]**

**Time and quality.** README: timestamps are UTC. **[verified]** Consistent with the files: the first bar of 2012 week 20 is Sunday 2012-05-13 21:01 and of 2020 week 10 is Sunday 2020-03-01 22:00, matching the 17:00 New York weekly open in summer and winter respectively. **[verified]** The README also says "Data points are indicative and based on the lowest spreads available exclusively on Active Trader accounts". **[verified]** Weekend handling: the 2020 week 10 file has 7,195 data rows and ends Friday 21:59, so closed periods appear to be omitted rather than filled. Within-week missing M1 bars were not audited. **[partly verified]** A bar timestamped 22:00 then 22:03 in the first rows of that file shows minutes without quotes are skipped, so M1 bars (and hence M5) will have holes in quiet periods. **[verified]**

**Licence.** README: "This is for personal use and abides by our EULA". **[verified]** The EULA page (`fxcm.com/uk/forms/eula/`) is a signing form; the clause text was not retrievable. Whether it permits local caching or ML training is **[unverified]**. FXCM tick data is not free (README: email premiumdata@fxcm.com). **[verified]**

## 2. Dukascopy

Primary sources: the Historical Data Export page (`dukascopy.com/swiss/english/marketwatch/historical/`), the Terms of Use (`dukascopy.com/swiss/english/legal-pages/terms-of-use/`), and direct requests to the `datafeed.dukascopy.com` host.

**Two access routes.**
1. Web export tool. The page says data is "in .csv format" in "different timeframes, from tick-by-tick to monthly". **[verified]** The page text does not list the bid/ask side option, timezone option, or any export size limit; the tool is script-driven and those controls were not visible to a plain page fetch. **[unverified]**
2. The `datafeed.dukascopy.com/datafeed/{PAIR}/{year}/{month0}/{day}/...` host used by third-party downloaders. It is not documented on the pages read; it is observed behaviour only. **[verified by request, undocumented]** Findings:
   - `BID_candles_min_1.bi5` and `ASK_candles_min_1.bi5` returned HTTP 200 for EURUSD 2024-03-10 (month is zero-based: `02` is March). Each is LZMA data of 1,440 records of 24 bytes (big-endian: seconds from midnight, open, close, low, high as integer price times 1e5, float volume). The bid and ask files are separate. **[verified]**
   - The day file is a full 24 hours of one-minute slots. Slots with the market closed or no quotes are present as flat bars with zero volume (for the file decoded, 1,261 of 1,440 slots were zero volume; the first non-zero slot was 21:00 UTC, the Sunday open). So gaps are filled with zero-volume placeholders that need filtering. **[verified]**
   - Timestamps in the day file are offsets in seconds from midnight UTC of the stated day (consistent with the 21:00 UTC Sunday open after the 2024-03-10 US DST change). **[verified by consistency, not by documentation]**
   - A request for `BID_candles_min_5.bi5` returned HTTP 429 then a connection timeout, so whether an M5 file exists was not determined. Earlier hour-1, tick and min_5 requests made in quick succession also returned HTTP 429 (body about 113 bytes, JSON), after which single requests worked again after a pause. So the host rate-limits bursts; the threshold is not documented. **[verified behaviour, M5 file existence unverified]**
   - Earliest EURUSD M1 day file found: 2003-05-02 (200). 2003-04-05/06/07 were weekend or timed out; earlier dates were not mapped. **[partly verified]**

**Volume.** The file's volume field is a float and nonzero only when quotes arrived. Whether it is tick count or notional is not documented in what was read. **[unverified]**

**Terms (verbatim from the Terms of Use page).**
- "The WEBSITE and the information contained therein may not be used to construct a database of any kind."
- "you may download material from the WEBSITE and/or make one print copy for your own personal, non-commercial use, provided that you retain all copyright and other proprietary notices."
- "You may not recirculate, redistribute or publish the analysis and presentation included in the WEBSITE without DUKASCOPY's prior written consent."
- A search-result summary of the same page also quotes a prohibition on any "scraper," "robot," "bot," ... to "access, acquire, copy, or monitor any portion of the WEBSITE" without consent; that sentence was not in the excerpt returned by my page fetch, so treat it as **[secondary]**.
- Whether the `datafeed` host counts as "the WEBSITE" is a legal reading that was not settled. **[unverified]**

## 3. HistData.com

Primary sources: `histdata.com/f-a-q/` and `histdata.com/f-a-q/data-files-detailed-specification/`, plus the EURUSD download page.

- Formats: tick and M1 only; no M5. **[verified]** (The EURUSD M1 page lists only M1.)
- Bid and ask: "Bar-based data (M1, 1-second bars) uses only the bid price"; tick data has both bid and ask. M1 columns: DateTime, Open, High, Low, Close, Volume (bid quotes). **[verified]**
- Timezone: "Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings adjustments" is what the FAQ and file specification state. **[verified]** Note ADR 0002 reports that for the tick files this statement does not hold (New York local with DST, then from 2019-03 a UTC-5 or Zurich-derived rule in some weeks). That is a contradiction between the site's own documentation and the observed tick data, found in this repo's earlier work. **[secondary: ADR 0002]**
- Depth: M1 EURUSD listed from 2000 through 2026-10 (full years to 2025, monthly files for 2026). **[verified]**
- Access: manual download per month or year; automatic delivery is a paid Google Drive subscription ($7/month per format) and FTP/SFTP is offered. **[verified]** The terms-of-use URL guessed for this research returned 404, and the FAQ states no redistribution terms ("at your own will and risk"). ADR 0002 records the same. **[verified for FAQ, otherwise secondary: ADR 0002]**

## 4. TrueFX

Primary sources: `truefx.com/` and the historical downloads page.

- Offers "top-of-book, tick-by-tick market data" for "16+ major currency pairs". **[verified]**
- Registration is required: "Please log in or register for free access to the historical data downloads." **[verified]**
- Start year, file format, rate limits and usage terms were not on the pages read (the pages point at a Market Data FAQ and a Historical Downloads section behind login). **[unverified]**
- ADR 0002 notes depth and terms are gated behind registration. **[secondary: ADR 0002]**

## 5. OANDA v20 (account-gated; not re-verified)

The OANDA developer documentation hosts tried (`developers.oanda.com`, `developer.oanda.com/rest-live-v20/instrument-ep/`) did not load (DNS failure and 404). Everything below is **[secondary: ADR 0002]**: bid/ask candles via the API from 2005, a broker (practice) account required, and licence terms that limit copies to installation and backup needs, require a log of copies, and require destruction of licensed materials on termination. The native M5 granularity and the bid/ask price selector are therefore **[unverified]** in this research. The ticket asked for free sources; a free practice account is the route ADR 0002 describes.

## 6. Free API tiers with a single price

- **Massive (formerly Polygon).** Forex aggregates endpoint built from "quoted bid/ask prices rather than executed trades"; volume is the count of quote updates; timezone UTC; limit up to 50,000 base aggregates per call. Currencies Basic (free): 5 API calls per minute, 2 years of history, "End-of-day" recency, individual use only. Starter ($49/month) has full history since 2009-09-25. **[verified]** Whether minute-level bars are available on the free plan was not stated on the pages read. **[unverified]**
- **Alpha Vantage.** `FX_INTRADAY` supports `5min` and is labelled "Premium" on the official documentation; the free key limit is "25 API requests per day". Response columns are timestamp, open, high, low, close (no bid/ask). **[verified]** History depth for `outputsize=full` is not stated on the page ("full-length intraday time series"). **[verified wording, depth unverified]**
- **Yahoo Finance (via yfinance).** The yfinance source limits 5m requests to the last 60 days (code comment `# 60 days`, 1m to 8 days, 1h to 730 days). yfinance's README states "the Yahoo! finance API is intended for personal use only." **[verified in the yfinance project's own code and README; Yahoo's own terms not fetched]** Far too short for a multi-year train/test split.
- **Twelve Data.** Basic plan lists "8 (800 a day)" API credits per minute (daily cap 800). Forex intraday depth on the free plan, bid/ask and terms were not on the page. **[verified limits, rest unverified]**
- **Tiingo.** Forex API documents top-of-book fields (`bidPrice`, `askPrice`, `midPrice`) and `resampleFreq=5min`. Free/Starter plan limits are 50 requests per hour, 1,000 per day, 1 GB per month, and the plan page does not list forex among its features. "Internal use" is defined as not displaying or sharing data. History depth and forex availability on the free plan are **[unverified]**.

## 7. Kaggle-hosted datasets

Kaggle metadata was read from Kaggle's own public dataset view/list API (no login). Kaggle requires an account to download files. Findings for EURUSD M5-relevant datasets:

| Dataset | What it is | Licence | Last updated | Notes |
|---|---|---|---|---|
| `stijnvanleeuwen/eurusd-forex-pair-15min-2002-2019` | Subtitle says 5 minute interval, slug says 15min, no description, ~77 MB | Unknown | 2020-12-21 | Interval ambiguous; provenance, price type unstated |
| `mmquants/eurusd-quotes-m5-20132017-mt4` | M5 quotes 2013 to 2017 from MT4, ~22 MB | Unknown | 2020-01-20 | No description; broker feed and timezone unstated |
| `cjdfgg/eurusd-m5-data` | ~85 MB, no description | CC0 | 2026-08-02 | Provenance, range and price type unstated |
| `imetomi/eur-usd-forex-pair-historical-data-2002-2019` | M1 and hourly bid and ask, 2002 to 2020, ~787 MB | AGPL-3.0 | 2020-12-02 | Description: "downloaded from Quantconnect.com", uploader offers to remove on request; ends 2020; not M5 |
| `orkadd/eurusd-5m-ohlcv-forex-historical-data` | "Free evaluation sample" of a commercial product; the full set is on ork.ad | MIT | 2026-07-06 | About 2 MB sample; single price plus volume |
| `getdatafinance/eurusd-5m-ohlcv-forex-historical-data` | Sample (about six months, 2026-03-26 to 2026-09-25) of getdata.finance | MIT | 2026-09-26 | Single price; full data elsewhere |
| `sentinelx/eurusd-5min-historical-data` | Sample 2025-10-02 to 2026-04-03; full 2008 to now sold at ork.ad | Apache-2.0 | 2026-04-08 | Promotional sample |

All of the above are **[verified]** as to the Kaggle metadata. The file contents themselves were not downloaded or inspected, so bid/ask presence, gaps, timezone and the true upstream source of the large undescribed sets are **[unverified]**. Most M5 sets show one price series only, and the licence field is the uploader's label, not necessarily the right of the underlying market-data vendor to license it.

## Cross-cutting points relevant to the map's stated shape

- **Bid/ask M5 availability.** No source examined publishes native M5 bars with both bid and ask. The sources that give both sides do so at M1 (FXCM in one file; Dukascopy in separate BID and ASK files) or at tick level (HistData, TrueFX, Dukascopy, FXCM on request), so M5 would be built by aggregation. The OANDA API is the only candidate that may serve bid/ask candles at M5 natively, which is **[unverified]** here.
- **Aggregating M1 to M5 for both sides.** For FXCM the file has bid and ask OHLC per minute, so M5 bid OHLC and ask OHLC can each be formed from their own columns. Quiet minutes missing from the file (observed) would make some M5 bars incomplete or absent.
- **Gaps and closed hours.** FXCM omits closed periods and also skips quiet minutes; Dukascopy fills with zero-volume flat bars. The two conventions treat the project glossary's "Gap" differently.
- **Timezones.** FXCM is UTC (documented, consistent with the data); the Dukascopy day files are seconds-from-midnight UTC (consistent, undocumented); HistData documents fixed EST but ADR 0002 observed otherwise.
- **Terms.** The only terms text quoted from an owner here are Dukascopy's (restrictive), FXCM's one-line README note (personal use, EULA), and the individual-use notes of Massive and Tiingo. HistData, TrueFX and OANDA terms rest on ADR 0002 or were not retrieved.

## Not verified / open

- FXCM EULA clause text (caching, ML training).
- Whether Dukascopy offers an `min_5` day file, its request-rate threshold, and the export tool's bid/ask and timezone options.
- TrueFX format, start year and terms (behind login).
- OANDA M5 bid/ask candle parameters and current licence text.
- Contents of any Kaggle file (bid/ask, gaps, timezone, upstream source).
- Massive free-plan access to minute aggregates; Tiingo and Twelve Data forex depth on free plans.
- Cause of the FXCM week 1 and 2024-2026 gaps, and whether FXCM backfills them.
