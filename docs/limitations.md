# Limitations and acceptance bar

This page describes the MVP honestly, so a reader can tell what its numbers do and do not mean. Terms are defined in [GLOSSARY.md](../GLOSSARY.md).

Every item below is a **scope boundary**, settled deliberately. None is a bug to file, and none should be "fixed" without revisiting the spec.

## Scope boundaries

1. **Marked to Mid, no fill simulation.** The Agent trades at Mid; there is no order book, latency or partial fill.
2. **A fixed notional Spread cost, not the observed Spread.** The Spread in History is indicative and frequently zero, so it is not treated as a cost. Spread cost is a constant set deliberately wider than the Spread seen in History.
3. **One pair.** EUR/USD only.
4. **One source.** The FXCM free candle repository, never spliced with another vendor. The vendor's own crossed and zero-Spread rows are kept unaltered; no cross-side check or clamping is applied.
5. **A single pinned date range.** History is pinned to 2016-01-04 to 2024-08-23 UTC, so the Split is stable and results are comparable to each other.
6. **One fixed Position size.** Short, flat or long, always one unit. No sizing and no leverage.
7. **No clock or calendar feature of any kind.** The Observation is Mid movement over a Window plus the current Position. Nothing else.
8. **A single seeded configuration, not a swept one.** One seed, one configuration.

## Known limitations of the design

These were each settled as limitations rather than decisions, and are stated so a reader does not try to fix them.

1. **The 2022 feed regime break.** Crossing goes 12.37% (2021), 0.03% (2022), then zero (2023, 2024). Zero-Spread rows go 50.62% (2021) to exactly zero from 2022. From 2022 the 1st percentile of close Spread is +0.10 pip, so no row has a Spread below one point. That is a feed or pipeline change imposing a minimum Spread, not a quality trend. At the roughly 70/15/15 Split it lands near the train/validation boundary, so **train sits in the pre-2022 regime and both validation and test in the post-2022 one**. It is a limitation rather than a blocker because prices are 5-decimal, so Mid is quantized to 0.05 pip, and old-regime bid-ask bounce at a 0.20 pip median Spread is on the order of 0.1 pip. Both are an order of magnitude below the 1.0 pip Spread cost the Agent must clear, against a 1.4 to 1.6 pip median m1 Bar range. No policy can monetise a 0.05 pip artifact while paying 1.0 pip per round trip.
2. **A Bar's Open is a verbatim copy of the previous Bar's Close.** Measured bit-exact: 0 of 135,770 consecutive m1 Bars disagree. This is why the Open channel is absent from the Observation.
3. **Carry is missing, and not neutrally so.** Overnight financing needs a rate-differential series the MVP has no source for. Across 2022 to 2024 a short-EUR policy would have earned carry while buy-and-hold paid it.
4. **Spread cost does not widen at rollover.** The affected hour runs a 1.7 to 3.5 pip median Spread and a 3.7 to 7.6 pip p95, against about 0.20 elsewhere, and it shifts between 21:00 and 22:00 UTC with US daylight saving; it is not a fixed UTC hour. The Agent pays 1.0 pip there like anywhere else. Week open is also wide: the first 30 minutes run a median 2.2 pips, p95 5.7, max 14.7.
5. **No mechanism detects pre-2022 overfitting.** Validation drives nothing, so nothing in a run would reveal it.
6. **Nothing enforces the evaluate-test-once discipline.** The test period is evaluated once per configuration and never used to choose one, but the written rule is the only guard on the headline number.
7. **A negative result is ambiguous.** It cannot distinguish "no learnable signal in M5 Mid returns" from "this particular configuration failed": one seed, one configuration, no validation gating. Never over-read a negative headline number as a finding about the market.
8. **Training/evaluation horizon mismatch.** The Agent trains on 288-Bar Episodes but is evaluated over roughly 1,440-Bar Segments. This is benign, because the policy is Markov on Window plus Position, but the Agent never learns the forced end-of-Segment close. Evaluation imposes that close externally and identically on all three policies, so no asymmetry is introduced.

## Figures that are arithmetic, not measurements

Derived from the pinned range, not counted over History:

- about 450.6 trading weeks x 1,440 M5 Bars is roughly 648,800 Bars
- train at 70% is roughly 454,200 Bars
- test at 15% is 97,323 Bars, about 67.6 Segments (matching the "about 68 weekend round trips" figure)

The **weekend-Gap distribution is unmeasured**. Weekend moves are *expected* to be small enough that the roughly 68 extra round trips buy-and-hold pays barely move any number, but that is an expectation. If a hold-through buy-and-hold variant is ever preferred, measure the weekend-Gap distribution first.

Throughput figures are a **ceiling, not a prediction**: the benchmark used a deliberately cheap `step()` returning random data, so a real `step()` with market-data indexing and PnL bookkeeping adds on top. Headroom is large (at most 5.6 minutes at 1M timesteps against a roughly 40-minute worst case at 5M), but treat about 6 minutes as no guarantee.

## Why `gap_threshold` is 4 hours

The Gap distribution is bimodal. In-week dropouts top out around 55 minutes, weekends run about 49 hours, and **nothing was observed between 2 and 49 hours**. Every threshold inside that corridor produces identical Segments on the pinned History, so the value cannot be justified by outcomes, only by margin. 4 hours is chosen deliberately low within the corridor rather than centred in it: spanning a hole corrupts Observations, whereas fragmenting Segments only costs data. A threshold just above the dropout maximum would split dropout Gaps into spurious Segment boundaries and force extra closes that all three policies would pay.

The emptiness of the corridor is an **observation about this data, not a law**. The Gap histogram in the `fetch-data` summary is the standing check: if it ever reports a Gap between 2 hours and about 49 hours, the threshold needs revisiting. The caution is earned: the corridor's original lower bound was an inherited, unverified 20-minute figure that direct measurement later refuted by about 3x.

## Acceptance bar

**Code implementing the requirements satisfies the spec.**

- No end-to-end run is an acceptance requirement.
- No recorded result file is a deliverable.
- Running the pipeline (`fetch-data`, `train`, `evaluate`) and reporting back on what it produces is the **user's** iteration step, deliberately downstream of delivery. The spec's job is to deliver the MVP's code, not to prove the MVP works.
