# Terms and open facts for candidate EUR/USD data sources

Research for issue #133 (map #128). Research only: no source is picked here. Date of all checks: 2026-10-20.

**Restart rule.** Nothing from before the map was used (no ADRs, no glossary or code on `main`, no older issues, no other `research/*` branches). The only prior material used is the map (#128) and the claims tagged verified in the #129 findings file.

**Tags.**
- **[verified]**: read directly from the owner's own text, or observed first-hand in a live request or a downloaded file.
- **[secondary]**: obtained through an intermediary (a search result snippet, or the WebFetch summariser, which paraphrases and may not quote exactly). Treat quotes tagged this way as unconfirmed wording.
- **[unverified]**: could not be checked. Not evidence.

**Method notes.** Downloads were kept in an isolated scratch directory and read with `python -I`. Requests were spaced 2 to 10 seconds apart. Dukascopy returned HTTP 429 once (see section 2), and I stopped hammering that host after it. OANDA and TrueFX downloads were not possible without an account; I did not create accounts.

---

## Summary table

| Source | M5 native? | Bid and ask | Terms found | Account |
|---|---|---|---|---|
| FXCM candle repo | No (M1, H1, D1 only, from #129) | Both, in M1 rows (from #129) | EULA text not retrievable; README says "personal use and abides by our EULA" | None for download |
| Dukascopy datafeed | No file found at the tested M5 paths; the website export tool offers 5-minute candles | Separate BID and ASK files at M1 | Website terms found; no separate datafeed terms found | None |
| OANDA v20 | Yes, `M5`, with `price=B`, `A`, or combinations | Yes by documentation (not live-tested) | API licence found (OANDA Corporation 2021 form) | Practice or live v20 account plus token |
| TrueFX | No (tick data) | Format not visible without login | Terms of Service found | Free registration |
| HistData | No (M1 and tick only) | M1 is bid only; tick has bid and ask | No terms of use found; only a "no warranty" statement | None |

---

## 1. Terms of use

### 1a. FXCM candle repository

**What the repository itself says** [verified, raw README at `raw.githubusercontent.com/fxcm/MarketData/master/README.md`]:

- URL pattern: `https://candledata.fxcorporate.com/{periodicity}/{instrument}/{year}/{int of week of year}.csv.gz`, periodicity listed as `m1, H1, D1`.
- README states the year range as 2017 to 2020 (#129 found files outside that range, so the README is stale).
- Quoted notes, verbatim: "Timestamps are in UTC"; "Data points are indicative and based on the lowest spreads available exclusively on Active Trader accounts"; "This is for personal use and abides by our [EULA](https://www.fxcm.com/uk/forms/eula/)"; "For more information, you may contact us: api@fxcm.com".
- GitHub reports `license: null` for the repository, so the repo has no software licence file that would cover the data.

**The EULA text could not be retrieved** [verified that it is unavailable]:

- `curl` to `fxcm.com/uk/forms/eula/` (and `/forms/eula/`, `/markets/forms/eula/`) returns HTTP 403, a Cloudflare "Just a moment..." challenge.
- WebFetch reached the page [secondary]. It reported a landing page titled "FXCM Software End User License Agreement" containing a risk warning and a button to an Adobe Sign widget (`fxcmus.na1.adobesign.com/public/esignWidget?wid=...`), with no agreement text. Fetching the Adobe Sign widget returned only a "Loading..." shell.
- A web search snippet [secondary] quoted what looks like the opening of the EULA: "Stratos Group International, LLC and its affiliates are willing to license the Software to you only upon the condition that you accept all of the terms contained in this End User License Agreement", defining Software as including "downloadable computer program, software application, trading platforms, APIs ... Software also includes any data, media, files ... as well as Stratos' proprietary streaming price feed", and License as "a limited, revocable, non-sublicenseable, non-exclusive, non-transferable right to use Software". I could not locate the page the snippet came from, so I could not read the grant, restrictions, or any clause on storing data. The snippet describes a trading-software EULA; whether the candle files fall under it is only implied by the README's link.

**Other FXCM text** [secondary, via WebFetch summariser]:

- `fxcm.com/uk/algorithmic-trading/market-data/` says "Email premiumdata@fxcm.com to try our free entry-level data solutions or to gain access to premium data" and gives no terms for free data.
- FXCM website terms (`/uk/legal/website-terms-and-conditions/`) contain, as paraphrased by the summariser, "you may not without FXCM's prior written permission alter, modify, reproduce, distribute or commercially exploit any materials". Those are website terms and may not govern the candle host.

**Answer to the ticket question (downloading, caching locally, ML training for personal use):**
- Downloading and local caching for personal use: the only statement is the README's "personal use". No clause on caching was retrievable. [unverified]
- ML training: no text found on it either way. [unverified]
- Open item: the EULA needs to be read in full, which probably requires a browser (Cloudflare challenge plus Adobe Sign), or a written answer from api@fxcm.com.

### 1b. Dukascopy

**Website terms** [verified, raw HTML of `https://www.dukascopy.com/swiss/english/legal-pages/terms-of-use/`, HTTP 200, footer "(c) 1998-2026"]. Definitions and scope, verbatim: the terms refer to "DUKASCOPY's website (the "WEBSITE")" and to "the TCU". Relevant clauses, verbatim:

> Except written agreement between you and DUKASCOPY, you agree to use the WEBSITE solely for your own non-commercial use and benefit, and not for resale or other transfer or disposition to, or use by or for the benefit of, any other person or entity. You agree not to use, transfer, distribute, or dispose of any information contained in the WEBSITE in any manner that could compete with the business of DUKASCOPY or any of its Suppliers. You may not copy, reproduce, recompile, decompile, disassemble, reverse engineer, distribute, publish, display, perform, modify, upload to, create derivative works from, transmit, or in any way exploit any part of the WEBSITE, except that you may download material from the WEBSITE and/or make one print copy for your own personal, non-commercial use, provided that you retain all copyright and other proprietary notices.

> The WEBSITE and the information contained therein may not be used to construct a database of any kind. Nor may the WEBSITE be stored (in its entirety or in any part) in databases for access by you or any third party or to distribute any database WEBSITEs containing all or part of the WEBSITE. You may not use the WEBSITE in any way to improve the quality of any data sold or contributed by you to any third party.

**Does it cover the datafeed host?** Not settled by the text.

- The definition is "DUKASCOPY's website", with no domain named in the passages I read. `datafeed.dukascopy.com` is a different hostname from `www.dukascopy.com`. Whether it is "the WEBSITE" is a question of interpretation. [unverified]
- The `datafeed.dukascopy.com` root (`/`) and `/robots.txt` timed out in my attempts, so I could not look for a notice on the host itself. [unverified]
- I found no separate datafeed terms. The Important Disclaimer page (`/legal-pages/important-disclaimer/`) [verified] speaks of "any of the data and documents that may be downloaded" in general terms, mainly about offers and risk, and contains no data licence. A search result [secondary] suggests terms may be shown inside the JForex platform's historical-data download prompt; I did not check JForex.
- The website export tool at `widgets.dukascopy.com/en/historical-data-export` is first-party and downloads a CSV from the browser. Its JavaScript bundle (read as untrusted text) shows it offers Minute candles of 1, 2, 3, 5, 10, 15, 30 or 45 minutes, Tick, Second and Hour options, and a BID or ASK selector. [verified from the shipped JS; the tool itself was not run.] The "construct a database" clause is the plainest reading for what that tool produces, since it sits on the website's own domain family.

**Answer (caching locally, ML training):** the website terms allow download "for your own personal, non-commercial use" but forbid constructing a database of any kind and forbid using the information in ways that could compete with Dukascopy. Whether a local file cache of downloaded candles is "a database" is not answered by the text. No clause mentions machine learning. [verified text; interpretation unverified]

### 1c. Terms side by side for personal-use ML training

| Source | Explicit permission found | Explicit restriction found | ML mentioned |
|---|---|---|---|
| FXCM | "personal use" (README) | EULA unread | No |
| Dukascopy | download for "personal, non-commercial use" | "may not be used to construct a database of any kind" | No |
| OANDA | use of Licensed Materials for four listed purposes | internal use only; no third-party dissemination | No |
| TrueFX | "own internal purpose of viewing and analyzing FX market data" | no distribution of prices | No |
| HistData | none | none found | No |

---

## 2. Dukascopy M5 on the datafeed, and the 429 threshold

Base: `https://datafeed.dukascopy.com/datafeed/EURUSD/<year>/<month-0-indexed>/<day>/<SIDE>_candles_<unit>_<n>.bi5` (month is zero-based; `02` is March). All observations [verified, live, 2026-10-20, User-Agent "Mozilla/5.0"].

| Request | Result |
|---|---|
| `2024/02/10/BID_candles_min_1.bi5` (2024-03-10) | 200, 3009 bytes; headers include `server: YANDA`, `last-modified: Mon, 11 Mar 2024 00:02:09 GMT` |
| `2024/02/11/BID_candles_min_1.bi5` | 200, 10899 bytes |
| `2023/05/14/BID_candles_min_1.bi5` and `ASK_candles_min_1.bi5` (2023-06-14) | 200 and 200 (12002 and 11908 bytes) |
| `2023/05/BID_candles_hour_1.bi5` (month level) | 200, 6407 bytes |
| `2023/BID_candles_day_1.bi5` (year level) | 200, 4579 bytes |
| `2023/05/14/BID_candles_min_5.bi5` | 404, 162 bytes |
| `2024/02/10/BID_candles_min_5.bi5` (twice) and `min_7` | 503, 107 bytes ("No server is available to handle this request") |
| `2024/02/10/BID_candles_min_15.bi5` and `min_60`, `foo_1` | 404, 3464 bytes |
| `2023/05/14/BID_candles_min_15.bi5` and month-level `2023/05/BID_candles_min_5.bi5` | timed out (curl code 000, no response); inconclusive |
| `2024/02/10/ASK_candles_min_1.bi5` and `2024/02/11/ASK_candles_min_1.bi5` | 503, 107 bytes (although the ASK file for 2023-06-14 returned 200) |

**Findings.**

- No 5-minute file was found on the datafeed. Candle files that exist at the paths I tried: daily `min_1`, monthly `hour_1`, yearly `day_1`. Every `min_5` attempt returned 404 or 503, never 200. This is evidence of absence at the tested paths, not proof that no M5 path exists anywhere. [verified for the paths tested]
- The 503 responses are not understood. The same `ASK_candles_min_1` path returns 200 for 2023-06-14 and 503 for 2024-03-10 and 2024-03-11, so 503 may mean "backend missing for this object" rather than a rate limit, or the host may be partly unhealthy. #129 recorded ASK as available for 2024-03-10; I could not reproduce that today. [unresolved]
- Rate: HTTP 429 was seen once, with body `429 ... content-length: 113, content-type: application/json`. It came on roughly the tenth request in about 25 seconds (about 0.4 requests per second; requests spaced 2 seconds apart, several of which were 404 or 503), on a bare `HEAD`. The next `GET` of a known-good file about 5 minutes later returned 200. I stopped probing after the 429, so **the threshold is not determined**; all I can say is that 0.4 requests per second was enough to trigger it once in a burst of about ten. No documented limit was found. [observed once; threshold unverified]
- Several requests later in the session hung or timed out (`000`), including the host root, so the host can also be slow.

---

## 3. OANDA v20

All from the official docs at `developer.oanda.com/rest-live-v20/` [verified, HTTP 200 pages saved and read], unless marked.

**Native M5 and price selector.**

- Endpoint: `GET /v3/accounts/{accountID}/instruments/{instrument}/candles`. "Rate Limit 120 requests per second".
- `granularity` query values include "M5 5 minute candlesticks, hour alignment" (also S5, S10, S15, S30, M1, M2, M4, M10, M15, M30, H1 and above).
- `price` query: "The Price component(s) to get candlestick data for. [default=M]". Type definition: "Can contain any combination of the characters "M" (midpoint candles) "B" (bid candles) and "A" (ask candles)." So `price=BA` is permitted by the docs. Response candles then carry `bid` and `ask` objects with `o,h,l,c`, plus `volume` ("The number of prices created during the time-range represented by the candlestick") and `complete`.
- `count` default 500, maximum 5000; "Count should not be specified if both the start and end parameters are provided". 5000 M5 bars is about 17 days of continuous quotes (about 3.5 trading weeks), so multi-year history needs many paged calls.
- `from`/`to`, `smooth` (default false), `includeFirst`, `dailyAlignment` (default 17), `alignmentTimezone` (default America/New_York), `weeklyAlignment` (default Friday). M5 uses hour alignment, so the daily and weekly alignment parameters do not apply to it.
- Not live-tested: no token. How far back M5 history goes is not stated in the docs I read. [unverified]
- Best-practices page recommends limiting new connections to "twice per second (2/s)" and using persistent connections.

**Practice account and token requirements.**

- Hosts [verified, Development Guide]: practice `https://api-fxpractice.oanda.com` ("recommended for testing with your fxTrade Practice account and your personal access token"), live `https://api-fxtrade.oanda.com`. Authentication is "Required" on both.
- Getting Started [verified]: "To use this API you must have a v20 trading account, which is available to all divisions except OANDA Global Markets and OANDA TMS BROKERS S.A." Steps: "Try a free demo account or open a live account"; then, "Log in to the Account Management Portal (AMP) on fxTrade and select "Manage API Access" under "My Services". Generate your personal access token immediately after you agree to the API license."
- Authentication page [verified]: "Authentication is required to access your live accounts." and tokens are Bearer tokens that grant access to all of the user's sub-accounts.
- Regional wrinkles [secondary, search snippets, not read at the source]: OANDA Japan expires demo accounts 30 days after opening, and states an API is not available to customers who registered only a demo account. These are other OANDA entities' pages; whether the user's division allows a demo-only API token is [unverified] and depends on where the user lives.

**Licence terms on stored copies.**

- The docs' "API License" is a per-entity PDF. I read the **OANDA Corporation** form dated 2021-04-06 (`oanda.com/assets/documents/230/OC_-_API_License_Agreement_2021-04-06.pdf`) [verified, text extracted]. A Canada form (`.../470/202107_OANDA_Canada_-_API_License_Agreement.pdf`) also exists and was not read; other entities will have other forms. Which one applies depends on the user's OANDA division. [unverified for the user's case]
- Definitions, verbatim: "FxTrade Practice" means "the OANDA proprietary computer game which simulates the execution of currency exchange transactions and maintains virtual currency exchange accounts"; "FXTrade Rates" means "real-time currency exchange rates"; "Licensed Materials" means "(i) the FXTrade API; (ii) the FXTrade Rate Feed; (iii) the FXTrade Rates; (iv) the Documentation; (v) any other materials provided by OANDA under this Agreement; and (vi) all upgrades, revisions, fixes, updates or enhancements".
- 3.1 grant: "a limited, non-exclusive, terminable license to use the Licensed Materials during the term of this Agreement, solely for the purposes of: (a) creating custom interfaces to the FXTrade System; (b) accessing the FXTrade Rate Feed in order to retrieve FXTrade Rates; (c) accessing FXTrade Accounts information, including account transaction history; and (d) creating automated trading systems for issuing Transactions to the FXTrade System or FxTrade Practice."
- 3.2: "(g) use the FXTrade System and the Licensed Materials for its own internal use only; (h) not permit any third party to use the FXTrade System or the Licensed Materials; and (i) not transmit, publish, disseminate, duplicate, display, disclose, offer or otherwise provide, in any form whatsoever, the FXTrade Rates to any third party." 3.2(a) limits copies "of the Licensed Materials" to those "necessary for Licensee's installation ... and to create back up copies".
- 15(a): licensee will not "communicate, resell or otherwise make available ... any Licensed Material (or any part thereof) to any third party, and that it will not re-transmit the same from its premises for any purpose".
- Termination (13): "the license to use the Licensed Materials will be immediately revoked" and "Licensee will forthwith destroy or return to OANDA all Licensed Materials and Proprietary Information, and in the event of destruction, Licensee will provide OANDA with documentation certifying such destruction". Either party may terminate on "no less than one (1) day written notice" (13.1).
- Fees: Schedule A says "OANDA has waived the license fee; a fee may be reinstated in the future by OANDA on 90 days' notice to the Licensee."
- The text has **no clause that mentions storing, caching, historical data, or machine learning**. Whether downloaded historical candles count as "FXTrade Rates" (defined as "real-time" rates) or as other "Licensed Materials", and so would have to be destroyed on termination, is an interpretation I did not settle. Personal-only, internal use is the clear pattern; a local cache used only by the account holder fits "internal use only", but the termination-destroy clause is the risk to weigh. [interpretation, unverified]
- The `oanda.com/site/terms-of-use` link in the docs footer returns 404. [verified]

---

## 4. TrueFX

- Operator: "TrueFX is a service brought to you by Integral" (FAQ); data "sourced from Integral OCX". [verified, `truefx.com/truefx-market-data-faq/`]
- Granularity: "historical tick-by-tick market data, at zero cost to you"; page headline "Top-of-book, tick-by-tick market data, with fractional pip spreads in millisecond detail". So tick data, no bars. [verified, `truefx.com/truefx-historical-downloads/`]
- Login: the downloads page says "Please log in or register for free access to the historical data downloads." The registration form (`/truefx-registration/`) shows "Price: Free", asks name, address, country, username, email and password, and requires ticking "I have read and agree to the Terms Of Service". [verified]
- **File format (columns, timestamp format and timezone, file layout) could not be seen**: the downloads sit behind the login and I did not register. A guessed file URL returned 404. [unverified]
- Terms of Service (`/truefx-terms-and-conditions/`, copyright "2009-2026") [verified, full text read]. Key sentences, verbatim:
  - "B. TrueFX Historical Data Download: This part of the System is designed to allow You to access and use historical FX datafor Your own internal purposes."
  - "Integral grants to You a non-exclusive, revocable, non-transferable ..., non-sublicensable and limited license for You to download a single instance of, and access, the "System," or parts thereof, for Your own internal purpose of viewing and analyzing FX market data. Without limiting the foregoing, You shall not transmit, distribute, publish or otherwise disseminate the System, or any part thereof, and/or the content on or from the System including, without limitation, FX prices."
  - "You will not (i) copy, alter, decompile or reverse engineer the System or any of its components, or (ii) sell, lease, transfer, make derivative works from the System."
  - "As of the Effective Date, there is no charge for the access and use of the System ... Integral may institute and/or modify a fee for access and use of the System upon no less than 45 days notice".
  - "INTEGRAL MAY STOP PRODUCING OR UPDATING ALL OR ANY PART OF THE SYSTEM AT ANY TIME WITH OR WITHOUT NOTICE." Termination: either party "with or without reason, by written notice".
  - Disputes go to American Arbitration Association arbitration in San Francisco.
  - The FAQ lists use cases of "enhanced price verification, historic back-testing and generating a base price for onward distribution", but the Terms contain the no-dissemination clause above.
- ML training / local caching: the terms say "internal purposes" and "viewing and analyzing", and do not mention caching or machine learning. A "derivative works" ban exists for the System; whether a trained model counts is not addressed. [verified text; interpretation unverified]
- Account details: name, address and country are collected at signup (fields appear optional in the form markup except name, username, email and password). [verified from form markup]

---

## 5. HistData, from the site and its files alone

Pages read: home, `/f-a-q/`, `/f-a-q/data-files-detailed-specification/`, `/about-us/`, `/download-by-ftp/`, `/privacy-policy-2/`, and the download pages for EURUSD. [verified]

**Formats and granularity** [verified, FAQ and specification]:
- "We can only deliver you time ordered Tick and M1 (1 minute) data. The data that we have available is organized by forex-pair/year/month and in zip format." No M5.
- Formats: Generic ASCII, Microsoft Excel, MetaTrader, MetaStock, NinjaTrader, in M1 bars; Generic ASCII in ticks; NinjaTrader also offers 1-second tick bars with LAST, BID or ASK quotes. The EURUSD download pages I read show the ASCII M1 and tick selections.
- Generic ASCII M1 row: `YYYYMMDD HHMMSS;Open;High;Low;Close;Volume`, semicolon separated, 6 decimal places, volume always 0 in the samples.
- Generic ASCII tick row: `YYYYMMDD HHMMSSNNN,Bid,Ask,Volume`, comma separated, millisecond resolution.
- Prices: "the bar prices: Open, High, Low, Close in the data files are based on the tick Bid price. ... The Ask price is only included in the Tick data of the Generic ASCII format." So M1 is bid only; bid and ask exist only in ticks.
- Volume: the FAQ says volume information was removed; every row I read has volume 0.
- Coverage on the EURUSD M1 page: years 2000 to 2025 as full-year zips, and 2026 January to October as monthly zips; "DataFiles Last Updated at: 2026-10-18 22:40:40".
- Download mechanism: the download page contains a form that POSTs `tk`, `date`, `datemonth`, `platform`, `timeframe` and `fxpair` to `/get.php`, which returned the zip (with a `Referer` header). Each month also has a `.txt` status report (gaps in seconds, since "We're measuring all the gaps bigger than 1 minute").
- Sample sizes: EURUSD M1 2025-07 zip 305 KB (33,039 rows), 2025-01 zip 291 KB (31,603 rows), tick 2025-07 zip 10.1 MB (1,555,775 rows, 60.7 MB unzipped).
- M1 files are **sparse**: they only contain minutes with at least one tick. A full January would be 44,640 minutes but contains 31,603 rows, and the weekend gap is absent. The status report lists many gaps of 60 to 170 seconds. So any aggregation to M5 must decide how to treat missing minutes.
- FTP/SFTP is a paid option ($27 USD one-off via PayPal, per the home page). "Automatic Updates" is $7 USD monthly. The web download is free. [verified]

**Timestamps and timezone** [verified from site text and file inspection]:
- Site claim, repeated in the FAQ and in every format of the specification: "Eastern Standard Time (EST) time-zone WITHOUT Day Light Savings adjustments".
- File observations (EURUSD M1, ASCII):

| File | First bar after each weekend gap | Last bar before the weekend |
|---|---|---|
| 2025-01 (northern winter) | Sunday 17:00 (e.g. `20250105 170000`) | Friday 16:59 |
| 2025-07 (northern summer) | Sunday 17:00 (e.g. `20250706 170000`) | Friday 16:59 |

  The tick files show the same pattern (`20250706 170017843` first tick after the weekend, `20250704 165901069` last).
- The FX week opens at 17:00 New York time in both seasons, which would be 22:00 UTC in winter and 21:00 UTC in summer. If the stamps were a fixed UTC-5 (EST with no DST), the July open would appear at 16:00. It appears at 17:00. This suggests that, in these files, timestamps follow New York wall-clock time including daylight saving, contradicting the site's stated convention. [inference from file observation]
- A second check using event timing: the largest one-minute range in the 2025-07 file on 2025-07-03 (the Thursday US jobs report day) is at `083000`, and in the 2025-01 file on 2025-01-10 at `083000`. The 30 July 2025 file shows its largest ranges at `143400` to `143700`. These match US releases at 08:30 and a 14:00 FOMC decision plus press conference in New York local time in both winter and summer. The release schedules are my recollection of well-known times, not checked against BLS or Federal Reserve pages. [inference, event times unverified]
- Not checked: the exact DST changeover weeks (March and November), which would settle whether the convention is America/New_York or something else, and older years (the convention may have varied by year). Do not assume one rule for the whole 2000 to 2026 range. [unverified]

**Terms** [verified that none are published]:
- The site has an About page ("These data is then shared for free to help you out testing your own/bought strategies and expert advisors"), a FAQ "Why Free?" answer, a cookies page and a privacy policy (`/privacy-policy-2/`, 4,794 characters, about cookies and data privacy). `/terms-of-service/` returns 404.
- The only statement about use: "Since it's free data, you'll not get from us any kind of warranty or certification. Use the data at your own will and risk." The home page also says of FTP: "we're not selling the data!!!".
- No licence, no permission statement, no restriction on redistribution, caching or ML. Absence of terms is not a licence grant; the legal position is unclear. [verified absence]

---

## Open items for the decision ticket

1. FXCM EULA: needs a browser read or an email to api@fxcm.com.
2. Dukascopy: whether the datafeed host is "the WEBSITE", and whether a local cache counts as "a database"; the 429 threshold; why ASK returns 503 for March 2024 days; JForex download prompt text.
3. OANDA: which division's licence applies to the user; how deep M5 history goes (needs a token to test); whether stored historical candles are "FXTrade Rates" or other "Licensed Materials" for the termination-destroy clause.
4. TrueFX: file format and timezone (needs a free account).
5. HistData: DST convention across years, and the legal status of data with no published terms.
