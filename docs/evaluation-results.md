# Evaluation results by commit

A running log of `evaluate` output, one entry per commit, newest first. Terms are defined in [GLOSSARY.md](../GLOSSARY.md); read [limitations.md](limitations.md) before interpreting any number (one seed, one configuration, so a difference between rows is a single sample, not a finding).

Every run is on the same pinned History and Split, with `SEED = 42`, `TOTAL_TIMESTEPS = 1_000_000` and the same 93,080 rewarded test Bars over 68 Segments.

## Cost regimes

Results are grouped by cost regime. A regime is defined by the cost rule and by what the Agent observes. Rows from different regimes are not comparable: a difference across regimes mixes the cost change with the Observation change and is not a finding.

| Regime | Cost rule | Spread view | Observation size | Entries |
|---|---|---|---|---|
| Fixed 1.0 pip (historical) | Fixed 1.0 pip per Position change | None | 289 | `4b214c0`, `b17f473` |
| Spread cost | `max(close Spread of Bar t, 1.0 pip)` per Position change, half charged per unit (#195) | Cost scalar, second after Position | 290 | `0d0eb5b` |

Buy-and-hold is listed once per regime, since its cost differs between regimes. It is identical across every entry within a regime.

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
