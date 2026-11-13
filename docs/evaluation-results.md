# Evaluation results by commit

A running log of `evaluate` output, one entry per commit, newest first. Terms are defined in [GLOSSARY.md](../GLOSSARY.md); read [limitations.md](limitations.md) before interpreting any number (one seed, one configuration, so a difference between rows is a single sample, not a finding).

Every run is on the same pinned History and Split, with `SEED = 42`, `TOTAL_TIMESTEPS = 1_000_000` and the same 93,080 rewarded test Bars over 68 Segments.

## Cost regimes

Results are grouped by cost regime. A regime is defined by the cost rule and by what the Agent observes. Rows from different regimes are not comparable: a difference across regimes mixes the cost change with the Observation change and is not a finding.

| Regime | Cost rule | Spread view | Observation size | Entries |
|---|---|---|---|---|
| Fixed 1.0 pip (historical) | Fixed 1.0 pip per Position change | None | 289 | `4b214c0`, `b17f473` |
| Spread cost | `max(close Spread of Bar t, 1.0 pip)` per Position change, half charged per unit (#195) | Cost scalar, second after Position | 290 | `0d0eb5b` |
| Spread cost, Time in position | Same as Spread cost | Cost scalar, second after Position; Time in position, third (#202) | 291 | `0c54efc` |

Buy-and-hold is listed once per regime, since its cost differs between regimes. It is identical across every entry within a regime. The Time in position regime shares the Spread cost rule, so its buy-and-hold matches the Spread cost regime's; the Observation differs, so the Agent rows are still not comparable across the two.

## How to add an entry

1. Check out the commit and run `uv run shadow-trader train`, then `uv run shadow-trader evaluate`.
   - The first entry of a regime must be retrained fresh at its commit. Do not reuse an existing model for it.
   - Later entries in a regime may reuse the model already in `data/model/` only if it is loadable against that commit's Observation shape; say so in the provenance.
2. Copy the `evaluate` table and the selected Checkpoint (from `data/model/learning_curve.csv`, the row with `selected` true) into a new row and a new section. Add the row under its regime in the Summary.
3. Record provenance: how the model was produced (fresh retrain or reused), the cost rule (for example fixed 1.0 pip, or `max(close Spread, 1.0 pip)`), and whether the Agent observes the Spread scalar.
4. The first entry of the Spread cost regime must also note the accepted confound: the cost rule and the Spread view changed together, with no ablation, so its result cannot be attributed to either alone.
5. Evaluate the test period once per commit; never use it to choose a configuration.

## Summary

Net and drawdown in pips; Sharpe is per-Bar, un-annualised. Random is the mean of 20 turnover-matched draws.

### Time in position regime: `max(close Spread, 1.0 pip)`, Agent observes the cost scalar and Time in position

| Commit | Change | Cost rule | Checkpoint | Agent net | Agent Sharpe | Agent max DD | Position changes | Random net | Random Sharpe | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `0c54efc` | Time in position (#202) | `max(close Spread, 1.0 pip)`, Spread view, Time in position | 102,400 | -26.85 | -0.0049 | 32.90 | 77 | -42.95 | -0.0009 | FAIL |

Buy-and-hold under this regime: net -81.90 pips (-0.74%), Sharpe -0.0004, max drawdown 843.20 pips (7.64%).

### Spread cost regime: `max(close Spread, 1.0 pip)`, Agent observes the cost scalar

| Commit | Change | Cost rule | Checkpoint | Agent net | Agent Sharpe | Agent max DD | Position changes | Random net | Random Sharpe | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `0d0eb5b` | Spread cost (#195) | `max(close Spread, 1.0 pip)`, Spread view | 563,200 | -10,554.95 | -0.0618 | 10,664.80 | 20,885 | -14,420.17 | -0.0748 | FAIL |

Buy-and-hold under this regime: net -81.90 pips (-0.74%), Sharpe -0.0004, max drawdown 843.20 pips (7.64%).

### Fixed 1.0 pip, no Spread view regime (historical)

Cost rule: fixed 1.0 pip per Position change. The Agent had no Spread view (Observation size 289). Kept for history; not comparable to the Spread cost regime.

| Commit | Change | Cost rule | Checkpoint | Agent net | Agent Sharpe | Agent max DD | Position changes | Random net | Random Sharpe | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `4b214c0` | Features (#181) | Fixed 1.0 pip, no Spread view | 716,800 | -12,632.80 | -0.0657 | 12,648.25 | 23,260 | -15,608.04 | -0.0808 | FAIL |
| `b17f473` | Checkpoints (#159) | Fixed 1.0 pip, no Spread view | 102,400 | -8,477.45 | -0.0658 | 8,493.10 | 16,091 | -10,618.69 | -0.0553 | FAIL |

Buy-and-hold under this regime, same in both: net -20.15 pips (-0.18%), Sharpe -0.0001, max drawdown 843.20 pips (7.64%).

## Time in position regime entries

## `0c54efc`: Time in position (#202)

Regime: `max(close Spread of Bar t, 1.0 pip)` per Position change, Agent observes the cost scalar and Time in position (Observation size 291).

Merge of PR #206.

| | net (pips) | net (% notional) | Sharpe | max DD (pips) | max DD (%) |
|---|---|---|---|---|---|
| Agent | -26.85 | -0.24 | -0.0049 | 32.90 | 0.30 |
| buy-and-hold | -81.90 | -0.74 | -0.0004 | 843.20 | 7.64 |
| random (mean) | -42.95 | -0.39 | -0.0009 | 210.29 | 1.91 |

| | gross (pips) | Spread cost (pips) | cost / change |
|---|---|---|---|
| Agent | 26.15 | 53.00 | 0.6883 |
| buy-and-hold | 47.85 | 129.75 | 0.9540 |
| random (mean) | 7.84 | 50.79 | 0.6595 |

- Selected Checkpoint: 102,400 steps. Validation there: net -76.50 pips, Sharpe -0.0044, 231 Position changes. The final Checkpoint (1,001,472) scored net -15,872.90 pips, Sharpe -0.0528, 28,373 changes on validation.
- Agent Position changes on test: 77.
- Agent beats 60% of random draws on net return and 10% on Sharpe.
- Verdict: **FAIL**. Sharpe not above buy-and-hold (-0.0049 vs -0.0004) and not above random (-0.0049 vs -0.0009). Net return is above buy-and-hold (-26.85 vs -81.90 pips) but that alone does not pass.
- Gross P&L is positive (26.15 pips) and cost (53.00 pips) turns it negative. Gross P&L is over 77 changes, so it is a single small sample.
- Provenance: fresh `uv run shadow-trader train` then `uv run shadow-trader evaluate` at this commit (clean tree), run by the user on 2026-11-13.
- The selected Checkpoint is a single noisy validation maximum (limitations 7-9), and the last Checkpoint of the same run traded 28,373 times on validation, so the run's behaviour varies widely across Checkpoints.
- The Agent traded far less than at `0d0eb5b` (77 vs 20,885 Position changes on test). Time in position is the only behavioural change between the two commits (`790138f` only adds reporting; net at `0d0eb5b` reproduces exactly), so the difference is the input's effect plus run-to-run variance. One seed cannot separate the two (limitations 7-8).
- Same cost rule and Spread view as `0d0eb5b`. It is listed as its own regime only because the Observation differs, per the regime definition above.

## Spread cost regime entries

## `0d0eb5b`: Spread cost (#195)

Regime: `max(close Spread of Bar t, 1.0 pip)` per Position change, Agent observes the cost scalar (Observation size 290).

Merge of PR #200.

| | net (pips) | net (% notional) | Sharpe | max DD (pips) | max DD (%) |
|---|---|---|---|---|---|
| Agent | -10,554.95 | -95.66 | -0.0618 | 10,664.80 | 96.66 |
| buy-and-hold | -81.90 | -0.74 | -0.0004 | 843.20 | 7.64 |
| random (mean) | -14,420.17 | -130.69 | -0.0748 | 14,459.13 | 131.05 |

- Selected Checkpoint: 563,200 steps. Validation there: net -15,387.35 pips, Sharpe -0.0531, 28,473 Position changes.
- Agent Position changes on test: 20,885.
- Agent beats 100% of random draws on net return and 100% on Sharpe.
- Verdict: **FAIL**. Net return and Sharpe not above buy-and-hold: -10,554.95 vs -81.90 pips, -0.0618 vs -0.0004.
- Provenance: model built by the user with `uv run shadow-trader train` (`data/model/` saved 2026-11-09, loadable against the 290-wide Observation); evaluated with `uv run shadow-trader evaluate` at this commit. The user's report that it is a fresh retrain at this commit was not independently verified.
- Accepted confound: the cost rule and the Spread view changed together, with no ablation, so this result cannot be attributed to either alone. It is also not comparable to the fixed 1.0 pip regime.

## Fixed 1.0 pip, no Spread view regime entries (historical)

## `4b214c0`: Features (#181)

Regime: fixed 1.0 pip, no Spread view.

Merge of PR #188. Adds the Feature families (return, SMA, volatility, range horizons) to the Observation.

| | net (pips) | net (% notional) | Sharpe | max DD (pips) | max DD (%) |
|---|---|---|---|---|---|
| Agent | -12,632.80 | -114.49 | -0.0657 | 12,648.25 | 114.63 |
| buy-and-hold | -20.15 | -0.18 | -0.0001 | 843.20 | 7.64 |
| random (mean) | -15,608.04 | -141.46 | -0.0808 | 15,641.80 | 141.77 |

- Selected Checkpoint: 716,800 steps. Validation there: net -15,033.20 pips, Sharpe -0.0489, 29,116 Position changes.
- Agent Position changes on test: 23,260.
- Agent beats 100% of random draws on net return and 100% on Sharpe.
- Verdict: **FAIL**. Net return and Sharpe not above buy-and-hold.
- Provenance: evaluated with the model already in `data/model/` (saved after this commit and loadable against its Observation shape), not retrained for this entry.

## `b17f473`: Checkpoints (#159)

Regime: fixed 1.0 pip, no Spread view.

Merge of PR #165, the commit before Features.

| | net (pips) | net (% notional) | Sharpe | max DD (pips) | max DD (%) |
|---|---|---|---|---|---|
| Agent | -8,477.45 | -76.83 | -0.0658 | 8,493.10 | 76.97 |
| buy-and-hold | -20.15 | -0.18 | -0.0001 | 843.20 | 7.64 |
| random (mean) | -10,618.69 | -96.24 | -0.0553 | 10,672.27 | 96.73 |

- Selected Checkpoint: 102,400 steps. Validation there: net -10,851.80 pips, Sharpe -0.0532, 22,466 Position changes.
- Agent Position changes on test: 16,091.
- Agent beats 100% of random draws on net return and 0% on Sharpe.
- Verdict: **FAIL**. Net return and Sharpe not above buy-and-hold; Sharpe not above random.
- Provenance: trained and evaluated fresh in a clean worktree at this commit.
