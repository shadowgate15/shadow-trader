# Shadow Trader

A reinforcement-learning system that learns to trade forex on short intraday bars, trained and evaluated offline first, then promoted to paper and eventually real-money trading.

## Language

**Bar**:
One fixed-length time interval of price data for an instrument. Length is configurable; 5 minutes is the default and 15 minutes the fallback.
_Avoid_: Candle, tick

**Instrument**:
A single tradable forex pair, such as EUR/USD.
_Avoid_: Symbol, ticker, asset

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
