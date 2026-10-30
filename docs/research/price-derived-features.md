# Research: price-derived features used in RL trading on short bars

Resolves #175 (parent map #174). Facts only; no recommendation for this repo. Each claim cites the source that owns it. Throwaway branch `research/price-derived-features`, never merged.

## Scope and honesty about coverage

Few primary sources describe price-only feature sets for RL on short forex bars. The sources actually read are below. Anything not read is listed under "Not verified".

| Ref | Source | What it is |
|---|---|---|
| [Z] | Zhang, Zohren, Roberts, "Deep Reinforcement Learning for Trading", arXiv:1911.10107 (https://arxiv.org/abs/1911.10107) | DQN/PG/A2C on 50 daily-frequency futures (incl. FX); 60-observation state |
| [G] | Goluža et al., "Deep reinforcement learning with positional context for intraday trading", arXiv:2406.08013 (https://arxiv.org/html/2406.08013v1) | PPO, MLP, 1-minute futures bars (commodities + FX futures) |
| [J] | Jiang, Xu, Liang, "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem", arXiv:1706.10059 (https://arxiv.org/abs/1706.10059) | 30-minute crypto bars; high/low/close tensor |
| [TA] | TA-Lib source: `ta_RSI.c`, `ta_ATR.c`, `ta_WILLR.c` (https://github.com/TA-Lib/ta-lib/tree/main/src/ta_func); function list https://ta-lib.org/functions/ | Reference implementations of the indicator definitions |
| [SB3] | Stable-Baselines3 RL tips (https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | Library guidance on observation normalisation |
| [PD] | pandas docs: `DataFrame.ewm` (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html), `DataFrame.rolling` (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) | Trailing-window / EWM semantics |

## 1. Feature families by source

### 1.1 Multi-horizon returns / momentum

- [G] (1-minute bars): returns over 1, 5, 15, 30 and 60 minutes, `r = (p_t^C - p_{t-w}^C) / p_{t-w}^C` (Sec. 3.1.1, Eq. 5). Normalised by standardisation using a running mean and std over a lookback (Table 3; the lookback is printed as "55 days" and may be a rendering artefact).
- [Z] (daily bars): returns over the past month, 2 months, 3 months and 1 year, normalised by volatility scaled to the horizon: "we normalise annual returns as `r_{t-252,t} / (sigma_t * sqrt(252))` where sigma_t is computed using an exponentially weighted moving standard deviation of r_t with a 60-day span" (Sec. 3, State Space).
- [TA]/ta-lib.org: ROC is listed as "((price/prevPrice)-1)*100", the same percent return over a chosen period.
- Pattern: horizons span roughly 1x to 60x the bar interval in [G], and 21x to 252x in [Z]. Both use close-to-close simple returns, then rescale by a volatility-like or running-statistics denominator.

### 1.2 Distance from moving averages / MACD-style trend

- [Z]: no plain "distance from SMA". It uses MACD built from exponentially weighted moving averages: `q_t = (m(S) - m(L)) / std(p_{t-63:t})`, then `MACD_t = q_t / std(q_{t-252:t})`, where `m(S)` is an EWMA of price with time scale S and std(p) is a 63-day rolling price std (Eq. 3). That is an MA difference divided by a price-volatility measure, then standardised again by its own 252-day std. The S, L pairs are not given in the state-space section.
- [G]: no moving-average or MACD feature in its list (returns, RSI, ADX, ULTOSC, Williams %R, plus positional features).
- [J]: no moving averages; the network sees normalised price history directly (1.5).
- Pattern: where an MA-based feature is used, the difference of two averages is divided by a volatility measure so it is unitless.

### 1.3 Volatility (stdev, ATR-style range)

- [Z]: volatility is used as a normaliser, not as a standalone input feature. sigma_t is an EW moving std of returns, 60-day span (also used in the reward). A 63-day rolling std of price normalises MACD.
- [G]: no volatility feature in the state.
- [TA] definition of ATR: true range is the largest of high-low and the distances from the prior close to the current high and low; it is averaged "using Wilder method", `ATR_t = ((n-1) * ATR_{t-1} + TR_t) / n`, default period 14, with "an unstable period comparable to an Exponential Moving Average" (`ta_ATR.c`). TA-Lib also lists NATR (normalised ATR) and STDDEV as separate functions (ta-lib.org/functions). ATR is in price units, so a unit-free input needs scaling (as NATR does); TA-Lib's source comments do not prescribe the scaling.
- No RL source read feeds ATR itself to the network (see "Not verified").

### 1.4 Position within a recent high-low range

- [G]: Williams %R, window 14, one of the price features, min-max scaled to [-1,1] (Table 3 groups RSI, ADX, ULTOSC, WILLR as "0-100 to [-1,1]"). TA-Lib's WILLR returns values in [-100, 0], `(highest - close) / (highest - lowest) * -100`, default period 14, and returns 0.0 when the window's high-low range is flat (`ta_WILLR.c`). How [G] maps WILLR's range onto the stated scaling is not stated in the text read.
- [G] also uses the Ultimate Oscillator (windows 7, 14, 28) and ADX (window 14).
- [Z], [J]: no explicit range-position feature. [J] feeds high and low prices normalised by the latest close (1.5).

### 1.5 Raw normalised price history (relative to the latest close)

- [J]: the input is a rank-3 tensor of shape (f, n, m): f = 3 features (close, high, low), n = 50 periods, m assets. "all prices in the input tensor will be normalization by the latest closing prices", i.e. `V_t = [v_{t-n+1} / v_t | ... | v_{t-1} / v_t | 1]`, likewise for high and low (Sec. 4.1, Eq. 18). The authors say this normalisation "is empirically better performing than others". Bars are 30 minutes; n = 50 is described as "a day and an hour". It is a per-Bar series re-based to the last Bar at each step.
- [Z]: "Normalised close price series" is also part of the state (Sec. 3); its normalisation is not spelled out in the text read.

### 1.6 Oscillators (RSI-style)

- [Z]: RSI "moving between 0 and 100", "oversold (a reading below 20) or overbought (above 80)", look-back window 30 days (Sec. 3). No scaling stated in the text read.
- [G]: RSI window 14 on 1-minute bars, min-max scaled from [0,100] to [-1,1] (Table 3).
- [TA] `ta_RSI.c`: Wilder smoothing of gains and losses ("Subsequent prevLoss and prevGain are smoothed using the previous values (Wilder's approach)"), equivalent form `RSI = 100 * prevGain / (prevGain + prevLoss)`, default period 14, a zero total (no gains and no losses) returns 50, and an unstable period before values converge.

## 2. Scaling and normalisation summary

| Feature | [G] 1-min bars | [Z] daily bars | [J] 30-min bars |
|---|---|---|---|
| Returns | running mean/std standardisation | divided by EW std of returns, scaled by sqrt(horizon) | prices divided by latest close (ratio) |
| Oscillators (RSI, ADX, ULTOSC, WILLR) | min-max to [-1,1] | RSI scaling not stated | none used |
| MA-based | none | MA diff / 63-bar price std, then / 252-bar std of that | none |
| Time left in day | min-max to [-1,1] | n/a | n/a |

General library guidance [SB3]: "you should always normalize the input to the agent (e.g. using `VecNormalize` for PPO/A2C)"; for custom environments, normalise the observation space when bounds are known, and "normalization is critical" for continuous-action algorithms. The page discusses clipping only for actions, not observations.

## 3. Time series per Bar versus current value

- [Z]: "At a given time step, we take the past 60 observations of each feature to form a single state." Every feature (price, returns, MACD, RSI) is a 60-step series.
- [J]: 50 periods of high/low/close per asset, fed to CNN/RNN/LSTM networks as a (3 x 50) map; the feature history is the state.
- [G]: the state is a single 13-dimensional vector per step into a 128-64 ReLU MLP (PPO). The lookbacks live inside the indicators (RSI window 14, returns 1 to 60 minutes) rather than in a stacked series. The text read does not describe stacking past bars.
- So the sources split: [Z], [J] feed per-bar histories; [G] feeds a current-value vector of multi-horizon indicators. None of the three compares the two feeding styles directly (not verified beyond the text read).

## 4. Lookahead and warm-up pitfalls stated by sources

- [G]: rolling-window evaluation: "a training period of one year followed by a testing period of four months"; "During each roll, the model is exclusively trained on data from the corresponding training set"; the last month of training is a separate validation set; execution slippage is taken at the next bar's Open, "not known at the time of decision-making". A one-hour lookback is required before trading begins (Sec. 4.2), i.e. features are computed from real prior history. Missing bars are filled with the last close.
- [J]: the backtest principle is "not knowing any 'future' market information"; selecting assets by future volume rank "will inevitably and indirectly pass future price information to the experiment" (Sec. 2.4 / 3.1). Normalising by the latest close uses only current and past bars.
- [PD] rolling: default is a trailing window with labels at the right edge (`center=False`); with an integer window `min_periods` defaults to the window size, so the first `window - 1` outputs are NaN. `center=True` would use future values. `ewm` with `adjust=True` computes `y_t` from `x_t ... x_0` only; `adjust=False` is the recursion `y_t = (1 - alpha) y_{t-1} + alpha x_t`; `alpha = 2 / (span + 1)`. `ewm` `min_periods` defaults to 0, so early values come from very few observations unless set.
- [TA]: RSI and ATR use Wilder (recursive) smoothing with a documented "unstable period": early outputs depend on where the series starts and converge only after enough prior bars. A flat window gives RSI 50 and WILLR 0.0 per the source comments.
- [Z] scales its reward by an ex-ante volatility estimate `sigma_{t-1}` (previous step), keeping the scaler causal.

## Not verified

- Tropmann-Frick and Tran, forex scalping with deep RL (IOS Press, doi:10.3233/FAIA220494): only the abstract was accessible (CNN over Gramian Angular Field images of technical indicators, regression head for order size). Its data, features and normalisation were not read; no claims are made from it.
- Wilder, "New Concepts in Technical Trading Systems" (1978), the original RSI/ATR definitions: not read; definitions here come from TA-Lib's implementation comments.
- FinRL (arXiv:2111.09395): abstract only.
- No primary RL source read feeds ATR/NATR or a plain distance-from-SMA as a feature, and none compares per-bar series against a current-value vector.
