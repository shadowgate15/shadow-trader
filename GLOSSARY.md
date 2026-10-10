# Shadow Trader

A reinforcement-learning system that learns to trade forex on short intraday bars, trained and evaluated offline first, then promoted to paper and eventually real-money trading.

## Language

**Bar**:
One fixed-length time interval of price data for an instrument, aligned to the UTC clock and identified by its open time. Length is configurable; 5 minutes is the default and 15 minutes the fallback. Prices are bid-side, with the spread carried alongside so the ask is always derivable.
_Avoid_: Candle, tick

**Instrument**:
A single tradable forex pair, such as EUR/USD.
_Avoid_: Symbol, ticker, asset

**Session**:
One continuous open-market stretch, from the weekly open (Sunday ~17:00 New York) to the weekly close (Friday ~17:00 New York). The daily 17:00 New York rollover is a marker inside a Session, not a boundary. Forex has no holiday calendar; the weekly closure is the only expected closure.
_Avoid_: Trading day, market hours

**Gap**:
A Bar missing from inside an open Session because no quotes arrived. Never filled at ingestion and always flagged. The weekly closure between Sessions is not a Gap.
_Avoid_: Missing data, hole

**Subsystem**:
A deployable piece of the system with its own charter and its own wayfinder map: market data ingestion, dataset preparation, trading environment, agent and training, evaluation, broker integration, live orchestration.
_Avoid_: Module, component, service

**Charter**:
A subsystem's statement of what it owns, what it explicitly does not, and its inputs and outputs.

**Execution model**:
The shared rules for spread, slippage, leverage, margin and position handling that both the simulated trading environment and live trading obey.
_Avoid_: Fill model, broker model

**Promotion path**:
The ordered stages (backtest, paper, real money) an agent moves through, and the evidence required to pass each gate.
_Avoid_: Deployment pipeline, rollout

**Dataset**:
An immutable, versioned collection of Bars (and, once prepared, features) for one or more Instruments at one bar size, described by a manifest recording its source, time range and lineage. A change to the data produces a new Dataset, never an edit. A trained model can always name the exact Dataset it saw.
_Avoid_: Data dump, table
