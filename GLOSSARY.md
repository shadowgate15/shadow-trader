# Shadow Trader

A reinforcement-learning forex trading bot. The MVP learns a trading policy from historical prices and is judged offline against simple baselines.

## Language

**Bar**:
One time-bucketed price record for the traded pair, holding an open, high, low and close for the bid side and for the ask side. The MVP uses five-minute bars.
_Avoid_: Candle, tick, row

**Mid**:
The price midway between a Bar's bid and ask. The single price series the Agent observes.
_Avoid_: Price, close, mid-market

**Spread**:
The ask minus the bid on a Bar. An indicative quantity rather than a price anyone could trade at: in History it is frequently zero and sometimes negative. Nothing in the MVP treats it as a cost; see Spread cost.
_Avoid_: Bid-ask, gap

**History**:
The fixed series of Bars the MVP trains on and is judged against: one pair, one source, one pinned date range.
_Avoid_: Dataset, data, prices, feed

**Agent**:
The learned policy that chooses an Action at each Bar.
_Avoid_: Bot, model, strategy

**Observation**:
Everything the Agent sees at a Bar: how Mid moved over a Window, together with its own current Position. Nothing else — no indicators, no clock, and no view of the Spread.
_Avoid_: State, features, input

**Window**:
The fixed-length run of consecutive Bars ending at the current Bar whose Mid movement the Observation carries. Never spans a large Gap.
_Avoid_: Lookback, buffer, context

**Action**:
The Agent's choice at a Bar: go short, go flat, or go long.
_Avoid_: Signal, order

**Position**:
The Agent's current exposure: short, flat, or long, always one fixed size.
_Avoid_: Trade, holding

**Spread cost**:
The cost charged whenever the Position changes: a fixed notional spread, not the Spread observed on the Bar. Half of it is charged for each unit of Position crossed, so a round trip and a direct reversal each pay one full spread. Set deliberately wider than the Spread seen in History, which is an indicative quote rather than a price anyone could trade at.
_Avoid_: Fee, commission, slippage

**Reward**:
The net profit or loss of one Bar after Spread cost, with no shaping.
_Avoid_: Score, return

**Episode**:
One fixed-length contiguous stretch of Bars the Agent trades through during training, starting at a random Bar and lying entirely within a single Segment.
_Avoid_: Run, session

**Split**:
The chronological division of History into train, validation, and test periods. The test period is evaluated once per configuration, and never used to choose one.
_Avoid_: Fold, partition

**Checkpoint**:
A point during training at which the deterministic Agent is scored on the validation period. The best-scoring Checkpoint that has changed Position at least once is the Agent that `evaluate` loads.
_Avoid_: Snapshot, save point

**Baseline**:
A simple policy the Agent must beat on the test period under the same Spread cost. There are two: buy-and-hold, which holds a long Position through each Segment and is flat over weekends; and random, which makes Actions at random while matching the Agent's own trading frequency, so the comparison is about timing rather than how often each trades.
_Avoid_: Benchmark

**Gap**:
A break in the Bar series, such as a weekend or missing data. Windows, Episodes and Segments never span a large Gap.
_Avoid_: Hole, missing data

**Segment**:
One contiguous stretch of Bars between large Gaps, in practice a trading week. Evaluation runs Segment by Segment, and every policy starts and ends flat in each. Distinct from an Episode, which is a training stretch starting at a random Bar.
_Avoid_: Session, week, run

**Warm-up**:
The Bars at the start of a Segment before a full Window exists. The Agent takes no Action there and no Reward accrues, so they are not judged.
_Avoid_: Burn-in, priming, ramp
