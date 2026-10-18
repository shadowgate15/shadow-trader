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
The ordered stages an agent moves through, and the evidence required to pass each gate: backtest, paper trading, real money at minimum size, then real money at scaled-up size. Every stage change needs a human sign-off.
_Avoid_: Deployment pipeline, rollout

**Stage**:
One step on the Promotion path: backtest, paper, real money (minimum size), or real money (scaled up).
_Avoid_: Phase, environment

**Gate**:
The evidence and human sign-off required to move a Candidate from one Stage to the next.
_Avoid_: Checkpoint, approval step

**Candidate**:
One frozen agent, identified by its model artefact, training configuration, seeds, code revision and the exact Dataset it saw. Any change produces a new Candidate that restarts at the first Stage.
_Avoid_: Model version, build

**Halt**:
An automatic flatten-and-stop of a Candidate when a safety limit is breached. Resuming needs human sign-off after a written post-mortem.
_Avoid_: Pause, crash

**Demotion**:
A human decision to return a Candidate to an earlier Stage after sustained underperformance or divergence from the simulation.
_Avoid_: Rollback

**Dataset**:
An immutable, versioned collection of Bars (and, once prepared, features) for one or more Instruments at one bar size, described by a manifest recording its source, time range and lineage. A change to the data produces a new Dataset, never an edit. A trained model can always name the exact Dataset it saw.
_Avoid_: Data dump, table

**Seed cohort**:
A Candidate together with sibling runs that share its training configuration and Dataset but differ in seed. Evaluated together at the backtest Gate to show consistency across seeds. Siblings are evidence only and are never promoted.
_Avoid_: Ensemble, seed sweep

**Revision**:
A numbered version of one emitted live Bar. A late or catch-up quote inside the repair window produces the next Revision of the same Bar; earlier Revisions are kept, never overwritten. The as-seen view is the Revision current at a given moment; the latest view is the highest Revision.
_Avoid_: Correction, patch

**Repair window**:
The period after a live Bar closes during which late or catch-up quotes may still produce a new Revision. Once it passes, the Bar is sealed and later quotes are dropped and counted.
_Avoid_: Grace period, lookback

**Feature spec**:
The versioned definition of every `feat_*` column a prepared Dataset carries: what each is computed from and over how many Bars it looks back. Each feature value for a Bar depends only on that Bar and earlier ones. A change to any definition produces a new version, never an edit. The volatility family is the sample std of 1-Bar returns over horizons 3 and up (a std of one return is undefined, so there is no 1-Bar volatility column) plus ATR over close from spec v4.
_Avoid_: Feature config, indicator list

**Warm-up**:
The stretch at the start of a Session, or of a Dataset, during which a history-dependent feature has not yet seen a full lookback inside it. Features are null throughout; a window never crosses a Session boundary. In live use, the extra Bars a recursive feature needs beyond its lookback before a Tail's last value matches the full-history value.
_Avoid_: Burn-in, priming

**Usable Bar**:
A prepared Bar that is priced, carries no distrusted quality flag, and has every history-dependent feature present. Everything else is unusable, and a Bar records why; a feature left null by a missing input (such as `tick_count`) is reported as `window_contains_gap`. Raw rows are never changed to express this.
_Avoid_: Valid Bar, clean Bar

**Embargo**:
The stretch of Bars cut from the end of an earlier split so that nothing used for training or selection on it can reach into the next split. Sized to the longest feature lookback plus the episode length, in Bars; the cut Bars are absent from that split's Dataset.
_Avoid_: Purge, gap

**Scaler**:
The train-only statistics that rescale features into model inputs, stored as one immutable artifact per build and cited by spec, policy and training Dataset. Prepared Datasets always hold unscaled features; scaling happens at use.
_Avoid_: Normalizer, preprocessor

**Tail**:
The most recent run of live Bars, long enough to cover the longest lookback plus Warm-up, from which live features for the latest Bar are computed with the same definitions as a batch build. A short Tail yields null features, not errors.
_Avoid_: Window, buffer

**Ledger**:
The reward-independent record of what the trading environment did over a run: each stepped Bar's equity and position, every fill, every closed lot, and each episode's start and end. Evaluation computes everything from it, and a scripted baseline produces one in the same form.
_Avoid_: Trade log, history

**Run summary**:
The counts and totals read straight off a Ledger for one run, and for each episode in it: Bars, decisions, fills, flagged events, costs, and starting and final equity. It holds no ratios, drawdowns or comparisons; those belong to evaluation.
_Avoid_: Metrics, performance report

**Episode**:
One run of the trading environment from a reset to its end, always inside a single Session. It starts flat on a usable Bar and ends either by margin closeout (terminated) or by a time-based end: a fixed window length, the Session's end, or the end of the Dataset (truncated).
_Avoid_: Game, rollout, run

**Closeout**:
The forced flattening of the whole position when margin cannot be held. The engine detects it at a Bar's close and flattens in the same step, at that same close, rather than at the next Bar's open (a recorded deviation from the Execution model's open-fill rule: there is no later Bar to wait for inside an Episode that ends on it). It is the only terminating Episode ending. A step's reward is floored numerically at a wiped-out account (equity at or below zero), a guard rather than shaping.
_Avoid_: Liquidation, margin call

**Decision Bar**:
A usable Bar at whose close the agent is asked for an action. Every other Bar inside an Episode (a Gap, or a Bar made non-usable by a distrusted flag or a Gap in its lookback) is stepped through without consulting the agent, so one agent step can span several Bars.
_Avoid_: Tick, turn
