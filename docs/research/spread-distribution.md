# Spread distribution in the source History

Research for issue #190 (parent map #189). Facts only; no recommendations.

Spread here is `ask - bid` in pips (1 pip = 0.0001). Negative means crossed (bid above ask); zero means bid equals ask.

## Method and data

- Data: the local FXCM weekly m1 cache (`data/raw/m1/EURUSD/<year>/<n>.csv.gz`, 454 files, fetched by `src/shadow_trader/fetch.py`) and the M5 Bars built from it (`data/bars/EURUSD_M5.parquet`, built by `src/shadow_trader/bars.py`). Both are untracked local files; the numbers below are reproducible by re-running `fetch-data`.
- Range: the pinned History, 2016-01-04 to 2024-08-23 UTC. 3,198,939 m1 rows; 631,881 M5 Bars (the 648,800 in `docs/limitations.md` is arithmetic, not a count).
- "Close spread" of a Bar is `ask_close - bid_close`. Bars are built only from complete five-minute buckets, so some hours have fewer Bars than the clock allows (e.g. 19,079 Bars in the 17:00 New York hour versus about 26,800 in a normal hour).
- Quantiles are over Bars (or m1 rows where stated). Prices are 5-decimal; observed close Spreads are almost entirely multiples of 0.1 pip.
- Analysis code was a throwaway pandas script outside the repo.
- Times: "rollover" below means the 17:00 America/New_York hour, which is 21:00 UTC in US summer time and 22:00 UTC in winter time.

## Which components exist per Bar

Each M5 Bar carries bid and ask each as open, high, low, close (eight prices), so four Spread components exist: `ask_open - bid_open`, `ask_high - bid_high`, `ask_low - bid_low`, `ask_close - bid_close` (`bars.py` SCHEMA).

- The raw m1 CSV header is `DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose`.
- Open is a copy of the previous close, per side. m1 rows one minute apart: bid open equals previous bid close in 100% of cases, ask in 99.99997%. M5 Bars: 98.97% (the shortfall is across Gaps and incomplete buckets, not disagreement on consecutive Bars).
- Therefore the open Spread is the previous Bar's close Spread. The open, close, high and low components are each aggregated per side separately: the high Spread is `ask_high - bid_high` and need not occur at the same instant on both sides.
- Overall across all Bars (all years):

| component | zero % | negative % | median | p90 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|---|---|
| open | 7.81 | 1.82 | 0.2 | 0.4 | 0.5 | 2.5 | -2.1 | 20.2 |
| high | 9.02 | 2.32 | 0.2 | 0.4 | 0.5 | 2.4 | -2.0 | 17.6 |
| low | 9.29 | 2.50 | 0.2 | 0.4 | 0.5 | 2.4 | -2.1 | 24.0 |
| close | 7.81 | 1.82 | 0.2 | 0.4 | 0.5 | 2.4 | -2.1 | 42.2 |

- Median of (high Spread - close Spread) and (low Spread - close Spread) is 0.0 in every year.
- The high and low components are zero or crossed more often than open/close in the years 2017 to 2021 (e.g. 2018: zero 8.5% close, 11.8% high, 13.5% low; negative 2.4% close, 3.9% high, 4.6% low), because extremes are taken per side. From 2023 all four components are within 0.01% of each other.
- Share of Bars where any of the four components is crossed, by year: 2016 2.20%, 2017 10.74%, 2018 11.35%, 2019 9.29%, 2020 16.07%, 2021 11.41%, 2022 1.12%, 2023 0.006%, 2024 0.004%. Any component zero: 4.69, 26.66, 32.83, 40.33, 48.56, 50.46, 4.52, 0.014, 0.019 %.
- Deepest crossing anywhere in m1 rows: 2.2 pips.

## Distribution by year (Bar close Spread, pips)

| year | Bars | zero % | negative % | mean | p1 | median | p90 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | 73,619 | 1.02 | 0.41 | 0.40 | 0.0 | 0.3 | 0.5 | 0.7 | 2.3 | -2.0 | 42.2 |
| 2017 | 73,366 | 7.99 | 2.68 | 0.24 | -0.1 | 0.2 | 0.4 | 0.5 | 2.0 | -0.8 | 15.4 |
| 2018 | 73,490 | 8.47 | 2.39 | 0.21 | -0.1 | 0.2 | 0.3 | 0.4 | 1.9 | -2.1 | 7.8 |
| 2019 | 72,907 | 13.13 | 2.52 | 0.19 | -0.1 | 0.1 | 0.3 | 0.4 | 1.8 | -1.0 | 17.6 |
| 2020 | 73,368 | 16.82 | 4.26 | 0.22 | -0.1 | 0.1 | 0.3 | 0.5 | 3.0 | -1.5 | 8.9 |
| 2021 | 72,520 | 18.31 | 3.14 | 0.19 | -0.1 | 0.1 | 0.3 | 0.3 | 2.6 | -1.1 | 9.0 |
| 2022 | 73,214 | 1.78 | 0.31 | 0.32 | 0.0 | 0.2 | 0.4 | 0.6 | 2.9 | -0.3 | 9.0 |
| 2023 | 71,694 | 0.004 | 0.003 | 0.29 | 0.1 | 0.2 | 0.4 | 0.5 | 2.4 | -0.1 | 9.0 |
| 2024 | 47,703 | 0.004 | 0.002 | 0.29 | 0.1 | 0.2 | 0.3 | 0.5 | 3.0 | -0.1 | 8.9 |

The same table at m1 row level (close Spread) gives the same picture: zero % 1.07, 8.03, 8.37, 13.19, 16.76, 18.12, 1.82, 0.009, 0.002 and negative % 0.37, 2.66, 2.32, 2.46, 4.26, 3.08, 0.33, 0.001, 0.001 for 2016 to 2024; median 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2.

Row-level "any of the four components crossed / zero" (the `count_row` definition in `bars.py`), by year, over m1 rows: crossed 1.68, 9.52, 9.68, 8.13, 13.95, 10.43, 1.09, 0.004, 0.003 %; zero 4.15, 25.27, 30.29, 37.36, 44.94, 47.05, 4.35, 0.023, 0.007 %. Overall 6.31% crossed, 22.40% zero.

### Stability within and across years

- The regime changes inside 2022, not at a year boundary. Weekly zero-Spread share of Bars is 17% to 27% in each week of January 2022 (weeks of 3 Jan to 24 Jan; negative 2% to 5%) and exactly 0 for every week from the week of 31 January 2022 onward. So the break falls between the end of the week of 24 January and the week of 31 January 2022. 2022 as a year therefore mixes both regimes (1.78% zero), which is why it reads as neither.
- After the break the Spread never touches zero except for isolated Bars: the last zero-or-crossed close Spread is 2024-07-01 17:00 UTC; the last crossed one is 2024-03-12 12:30 UTC. Post-break p1 of close Spread is +0.1 pip; the post-break m1 minimum is -1.5 pips (in January 2022 data before the break; from 2023 onward -0.4 at most).
- Before the break, zero share drifts: quarterly (excluding rollover and week open) 2016 about 0.4% to 1.4%, 2017Q1 3.6%, 2017Q4 16.9%, 2018Q4 5.6%, 2019Q4 20.8%, 2021Q4 23.3%; negative share 0.1% to 6.0% across quarters. Median fell from 0.3 (2016) to 0.1 (2019 to 2021) and returned to 0.2 from 2022.
- Post-break quarterly median is 0.2 (0.3 in 2022Q1 and 2022Q2 only) and p95 0.3 to 0.5.
- Excluding the rollover hour and the first 30 minutes of the week, close Spread p99 by year: 1.2 (2016), 1.0, 0.6, 0.5, 0.9, 0.5, 1.0 (2022), 0.7 (2023), 0.6 (2024). With them included p99 is 1.8 to 3.0, so the upper tail is dominated by rollover and week open.
- Most common values (excluding rollover/week open): pre-2022 0.1 (129,668 Bars), 0.2 (128,123), 0.3, 0.0 (47,971), 0.4, -0.1 (9,331). Post-2022: 0.2 (76,630), 0.1 (45,104), 0.3 (41,145), 0.4 (14,985), 0.5, 0.0 (1,306, all but a handful in January 2022).
- Share of Bars with close Spread of at least 1 pip: 2.80% overall; 0.82% excluding the rollover hour. By year 2.3% to 3.5%, no trend.
- Day of week (excluding rollover/week open): Monday to Friday medians 0.2, p95 0.4 to 0.5, zero 7.9% to 8.4%; Sunday Bars (the week-open side not covered by the 30-minute exclusion) median 0.3, p95 0.9.

## By hour of day (Bar close Spread, UTC)

Outside 20:00 to 23:00 UTC every hour has median 0.2 (0.1 in some pre-2022 hours 10:00, 11:00, 15:00, 16:00), p95 0.4 to 0.5 and p99 0.5 to 0.7. Exceptions: 12:00 and 13:00 UTC have p99 1.3 and 1.0 over all years (1.3 and 0.8 pre-2022, 1.4 and 1.2 post-2022).

All years:

| UTC hour | zero % | neg % | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 20 | 6.37 | 1.83 | 0.2 | 0.5 | 0.9 | 4.8 | 42.2 |
| 21 | 2.91 | 1.30 | 0.7 | 3.2 | 4.3 | 6.8 | 15.4 |
| 22 | 1.01 | 0.38 | 0.4 | 1.9 | 2.8 | 5.1 | 17.6 |
| 23 | 3.62 | 0.73 | 0.2 | 0.5 | 0.6 | 0.9 | 8.2 |

Pre-2022 zero share by hour is 8.9% to 14.1% in hours 0 to 20 (3.9% at 21:00, 1.5% at 22:00, 5.2% at 23:00); post-2022 it is 0.6% to 0.9% in hours 0 to 20 and under 0.5% at 21:00 to 23:00.

### Rollover (17:00 New York hour)

By NY-local hour 17 (Bars): n 19,079; zero 0.34%; negative 0.45%; median 1.5; p90 3.6; p95 4.6; p99 6.7; max 17.6.

By year (Bar close Spread in the NY 17 hour): median / p95 / p99:

| year | median | p95 | p99 |
|---|---|---|---|
| 2016 | 1.1 | 3.9 | 5.7 |
| 2017 | 0.8 | 3.9 | 6.8 |
| 2018 | 0.9 | 3.3 | 5.0 |
| 2019 | 1.1 | 3.7 | 5.2 |
| 2020 | 2.2 | 5.3 | 6.9 |
| 2021 | 2.1 | 5.7 | 7.8 |
| 2022 | 2.0 | 5.6 | 7.4 |
| 2023 | 1.5 | 5.3 | 7.7 |
| 2024 | 2.2 | 4.9 | 6.3 |

At m1 row level the same hour gives median 0.8 to 2.4 and p95 3.6 to 6.2 per year (max 22.7). The rollover widening exists in every year; its level roughly doubles from 2020.

The hour shifts with US daylight saving: the UTC 21:00 hour has median 1.5 and p95 4.8 in summer-time Bars versus 0.2 and 2.0 in winter-time Bars; the UTC 22:00 hour has median 1.3 and p95 4.4 in winter-time Bars versus 0.3 and 0.8 in summer-time Bars.

The hour is also thinly populated: only 19,079 Bars (about 71% of a full hour's Bars) exist in it.

### Week open

Bars in the first 30 minutes after a break of more than 40 hours (weekend open), 1,897 Bars (5-minute buckets starting 0 to 25 minutes after the first Bar of the week):

- zero 0.05%, negative 0.16%, median 2.3, p90 4.9, p95 5.7, p99 8.2, min -2.0, max 15.4.
- Pre-2022: median 2.1, p95 5.5; post-2022: median 2.8, p95 6.4, max 9.0.
- By minute offset from week open: first 5 minutes median 2.7, p95 6.6, max 14.8; 25 to 30 minutes median 2.0, p95 5.0. It decays slowly across the 30 minutes.
- By year, median 1.6 to 3.0, p95 4.4 to 7.3.
- At m1 row level for the first 30 minutes: median 2.8, p95 7.0, p99 8.8, max 18.3.

## Effect of excluding rollover and week open

Excluding the NY 17:00 hour and the first 30 minutes of the week, close Spread by year (Bars): median 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2; p95 0.6, 0.4, 0.3, 0.3, 0.4, 0.3, 0.5, 0.4, 0.3. m1-row-level non-rollover p95 by year: 0.6, 0.4, 0.3, 0.3, 0.4, 0.3, 0.5, 0.4, 0.3.

## Reconciliation with `docs/limitations.md`

Reproduced:
- Regime break at the start of 2022 with zero Spread afterwards and crossing near zero afterwards: the post-break share is 0.004% zero and 0.003% crossed in 2023 (a few Bars per year), not exactly zero.
- Open equals previous Close; rollover Spread much wider than the rest; week open wide (first 30 minutes median about 2.2 to 2.3, p95 5.7, max 15.4 on Bars; the doc says max 14.7).
- Post-2022 p1 of close Spread is +0.10 pip.
- Normal-hour median about 0.20 pip.

Not reproduced under the definitions I could identify:
- The doc gives crossing 12.37% (2021), 0.03% (2022), zero in 2023 and 2024; zero-Spread 50.62% in 2021. Measured on any-of-four components: Bars 11.41% crossed and 50.46% zero in 2021 (m1 rows: 10.43% and 47.05%); 2022 is 1.12% crossed (m1 1.09%), not 0.03%; 2023/2024 are 0.006%/0.004% crossed, not exactly zero.
- "Zero-Spread rows go ... to exactly zero from 2022": 2022 has 4.5% Bars with a zero component, all concentrated before the week of 31 January 2022; 2023 and 2024 are 0.014% and 0.019%.
- Rollover hour "1.7 to 3.5 pip median, 3.7 to 7.6 pip p95": measured per-year NY-17 median is 0.8 to 2.2 (Bars) and p95 3.3 to 5.7. I did not identify which slicing produces the doc's ranges.
- Pre-break zero/crossed shares rise from 2016 to 2021 (see above) and fall to near zero within one week.

## What FXCM documentation says

Primary sources fetched 2026-11-04:

1. FXCM market data repository README, github.com/fxcm/MarketData (https://github.com/fxcm/MarketData; raw README https://raw.githubusercontent.com/fxcm/MarketData/master/README.md):
   - URL structure `https://candledata.fxcorporate.com/{periodicity}/{instrument}/{year}/{int of week of year}.csv.gz`, periodicity m1, H1, D1; week 1 to 52/53 for m1 and H1.
   - "Data points are indicative and based on the lowest spreads available exclusively on Active Trader accounts".
   - "Timestamps are in UTC".
   - The README does not use the words "bid" or "ask" and does not say whether the quotes are executable. The text gives no spread figure.
2. FXCM Forex and CFD Market Data page (https://www.fxcm.com/uk/trading-services/market-data/): "Our FX and CFD price feeds are aggregated in real-time to display the best available bid and offer per symbol"; "All of FXCM's market data solutions are based on executable pricing and real client trading behaviour" (presented as "more than indicative data"). This is marketing copy for FXCM's market-data products in general; it does not name the free candle repository.
3. FXCM account page (https://www.fxcm.com/markets/accounts/trade-now/): "Average low spreads for EUR/USD of 1.3" pips, "derived from tradable prices at FXCM", "at the date of May 17, 2023". The page does not name an account type for the figure.
4. FXCM Spread Costs page (https://www.fxcm.com/markets/accounts/spread-costs/): "Spreads are variable and are subject to delay"; "The spread figures are for informational purposes only"; Active Traders get discounts on spread costs and commission-based pricing applies to Active Trader accounts. The fetched copy's spread table and last-updated date were unpopulated template placeholders, so no EUR/USD value was recoverable there.

Statements by FXCM relevant to the tradeability question, side by side:
- Free repository: "indicative", based on the lowest spreads available exclusively on Active Trader accounts.
- Main market-data page and account page: pricing "executable" / "tradable".
- Neither a published Spread for the repository's own data nor a statement of whether its quotes were executable was found.

Comparison, measured versus published: the repository's median close Spread is 0.1 to 0.3 pip by year (0.2 post-break) against FXCM's published "average low spread" of 1.3 pips for EUR/USD (May 2023). Measured post-break mean close Spread is 0.29 pip.

## Not found

- No primary-source literature (academic or regulator) on whether free-repository bid/ask quotes are tradeable was reviewed; only FXCM documentation was consulted.
- The typical Spread a retail EUR/USD trader pays: the only primary figure found is FXCM's own 1.3 pip average low spread (May 2023, account type unnamed). No independent or regulator-published figure was gathered.
- The cause of the early-2022 change was not found in any FXCM source.
