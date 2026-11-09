# Limitations and acceptance bar

This page describes the MVP honestly, so a reader can tell what its numbers do and do not mean. Terms are defined in [GLOSSARY.md](../GLOSSARY.md).

Every item below is a **scope boundary**, settled deliberately. None is a bug to file, and none should be "fixed" without revisiting the spec.

## Scope boundaries

1. **Marked to Mid, no fill simulation.** The Agent trades at Mid; there is no order book, latency or partial fill.
2. **Spread cost is the observed Spread with a floor, and the quotes are indicative.** Each Position change pays half of `max(close Spread of the decision Bar, 1.0 pip)` per unit crossed (`SPREAD_PIPS` is the floor). The floor exists because FXCM calls the repository's quotes indicative and tradeability is unconfirmed: the median close Spread is 0.1 to 0.3 pip by year, against a published EUR/USD average low spread of 1.3 pips (May 2023), and 97.2% of Bars sit at the floor. So the cost is the floor on almost every Bar and follows the data only on the wide ones. Slippage, commission and market impact are not modelled. Several figures first recorded in this document did not reproduce in the Spread-distribution research (`docs/research/spread-distribution.md` on branch `research/spread-distribution`); see Known limitations 1 and 4.
3. **One pair.** EUR/USD only.
4. **One source.** The FXCM free repository of weekly m1 files, never spliced with another vendor. The vendor's own crossed and zero-Spread rows are kept unaltered; no cross-side check or clamping is applied.
5. **A single pinned date range.** History is pinned to 2016-01-04 to 2024-08-23 UTC, so the Split is stable and results are comparable to each other.
6. **One fixed Position size.** Short, flat or long, always one unit. No sizing and no leverage.
7. **No clock or calendar feature of any kind.** The Observation is Mid movement over a Window, plus Features derived from Mid, plus the current Position, plus the current Bar's Spread cost as one number. Nothing else.
   Features include the weekend jump: they look back across Gaps by Bar count, so early-week values reflect the prior week. The first Segment of History is used only as lookback and is dropped from training and evaluation.
8. **A single seeded configuration, not a swept one.** One seed, one configuration.

## Known limitations of the design

These were each settled as limitations rather than decisions, and are stated so a reader does not try to fix them.

1. **The 2022 feed regime break.** The break is real but the earlier figures did not reproduce. Per the research, the share of Bars with any component crossed is 11.41% (2021), 1.12% (2022), then 0.006% (2023) and 0.004% (2024), not zero; Bars with any zero component are 50.46% (2021), 4.52% (2022, all before the week of 31 January 2022), then about 0.01 to 0.02%. From 2022 the 1st percentile of close Spread is +0.10 pip, so no row has a Spread below one point. That is a feed or pipeline change imposing a minimum Spread, not a quality trend. At the roughly 70/15/15 Split it lands near the train/validation boundary, so **train sits in the pre-2022 regime and both validation and test in the post-2022 one**. It is a limitation rather than a blocker because prices are 5-decimal, so Mid is quantized to 0.05 pip, and old-regime bid-ask bounce at a 0.20 pip median Spread is on the order of 0.1 pip. Both are an order of magnitude below the 1.0 pip floor on the Spread cost the Agent must clear, against a 1.4 to 1.6 pip median m1 Bar range. No policy can monetise a 0.05 pip artifact while paying 1.0 pip per round trip.
2. **A Bar's Open is a verbatim copy of the previous Bar's Close.** Measured bit-exact: 0 of 135,770 consecutive m1 Bars disagree. This is why the Open channel is absent from the Observation.
3. **Carry is missing, and not neutrally so.** Overnight financing needs a rate-differential series the MVP has no source for. Across 2022 to 2024 a short-EUR policy would have earned carry while buy-and-hold paid it.
4. **Rollover and week open are costly but noisy.** Spread cost now widens there, because it follows the Spread above the floor. The rollover hour (17:00 New York, so 21:00 UTC in US summer time and 22:00 in winter, not a fixed UTC hour) measured a per-year median of 0.8 to 2.2 pips and p95 of 3.3 to 5.7 on Bars, against about 0.20 elsewhere; the earlier 1.7 to 3.5 median and 3.7 to 7.6 p95 did not reproduce. Week open is wide too: the first 30 minutes run a median 2.3 pips, p95 5.7, max 15.4 on Bars (the earlier max of 14.7 did not reproduce). The hour is thinly populated, and because the indicative feed may not be tradeable there, the cost charged at these Bars is itself uncertain.
5. **Validation selects a Checkpoint but does not detect pre-2022 overfitting.** Validation scores each Checkpoint and the best one is the Agent that `evaluate` loads, but nothing compares validation against train, so a policy that has memorised the pre-2022 regime is not flagged by it. Validation sits in the post-2022 regime, so it can disfavour such a Checkpoint only incidentally.
6. **Nothing enforces the evaluate-test-once discipline.** The test period is evaluated once per configuration and never used to choose one, but the written rule is the only guard on the headline number.
7. **A negative result is ambiguous.** It cannot distinguish "no learnable signal in M5 Mid returns" from "this particular configuration failed": one seed, one configuration, and a Checkpoint chosen on a single noisy validation score. Never over-read a negative headline number as a finding about the market.
8. **One seed means the learning curve is one sample.** The curve and the selected Checkpoint come from a single run. A different seed would give a different curve and possibly a different selection, so do not read its shape as a property of the method.
9. **The best Checkpoint's validation score is optimistically biased.** Picking the maximum of about 20 noisy Checkpoint scores selects partly on luck, so the winner's validation figure overstates what it would score on fresh data. The test figure stays clean: the test period is evaluated once and never used to choose, so selection never touches it.
10. **Training/evaluation horizon mismatch.** The Agent trains on 288-Bar Episodes but is evaluated over roughly 1,440-Bar Segments. This is benign, because the policy is Markov on Window plus Position, but the Agent never learns the forced end-of-Segment close. Evaluation imposes that close externally and identically on all three policies, so no asymmetry is introduced.

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
