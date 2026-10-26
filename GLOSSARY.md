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
The ask minus the bid on a Bar. An observed property of the data, not a cost model; see Spread cost.
_Avoid_: Bid-ask, gap

**History**:
The fixed series of Bars the MVP trains on and is judged against: one pair, one source, one pinned date range.
_Avoid_: Dataset, data, prices, feed

**Agent**:
The learned policy that chooses an Action at each Bar.
_Avoid_: Bot, model, strategy

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
One contiguous stretch of Bars the Agent trades through during training, starting at a random Bar.
_Avoid_: Run, session

**Split**:
The chronological division of History into train, validation, and test periods. The test period is evaluated once.
_Avoid_: Fold, partition

**Baseline**:
A trivial policy the Agent must beat on the test period under the same Spread cost: buy-and-hold, or random Actions.
_Avoid_: Benchmark

**Gap**:
A break in the Bar series, such as a weekend or missing data. Windows and Episodes never span a large Gap.
_Avoid_: Hole, missing data
