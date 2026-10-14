# Leakage-free time-series split and embargo practice (issue #44)

Question: what is established practice for leakage-free chronological splits on financial time series (single split vs walk-forward, purging and embargo sizing, train-only scaling), and which pitfalls matter for RL on intraday forex Bars? See issue #44, map #42, charter in #7. Researched 2026-10-14 (all "accessed" dates 2026-10-14). Throwaway research branch, never merged. Facts only; no choice is made here.

Evidence labels: **[V]** read on the owning primary page this session; **[S]** described by a secondary page (primary not readable this session); **[U]** from memory or not verified. Nothing marked [S] or [U] should be relied on without a spot check.

## 1. Why random splitting is invalid on time series

- scikit-learn: classical `KFold` and `ShuffleSplit` "assume the samples are independent and identically distributed, and would result in unreasonable correlation between training and testing instances (yielding poor estimates of generalization error) on time series data." Time series have autocorrelation between observations near in time. [V] https://scikit-learn.org/stable/modules/cross_validation.html (Time Series Split)
- Bailey, Borwein, Lopez de Prado, Zhu (2015, J. Computational Finance), abstract: "Standard statistical techniques designed to prevent regression overfitting, such as hold-out, tend to be unreliable and inaccurate in the context of investment backtests." They propose combinatorially symmetric cross-validation (CSCV) to estimate the probability of backtest overfitting. [V] https://scholarworks.wmich.edu/math_pubs/42 (SSRN: https://ssrn.com/abstract=2326253, 403 to fetcher)

## 2. Single split vs walk-forward vs blocked/purged CV

**Single chronological hold-out (out-of-sample, OOS).** Train on the earlier part, evaluate on the last part. Bergmeir, Hyndman, Koo (2018, Comput. Stat. & Data Anal. 120:70-83): K-fold CV is "not straightforward and often omitted by practitioners in favor of an out-of-sample (OOS) evaluation" because of serial correlation and potential non-stationarity; for the (purely autoregressive) setups they study, K-fold CV is argued valid and "performs favourably compared to both OOS evaluation and other time-series-specific techniques". [S: abstract via search result; the paper itself was not fetched] See also https://bridges.monash.edu/articles/journal_contribution/A_Note_on_the_Validity_of_Cross-Validation_for_Evaluating_Time_Series_Prediction/21521274/1 (certificate error from fetcher).
Scope caveat: that result concerns autoregressive forecasting with lagged inputs as features; it does not address overlapping forward labels or RL episodes. [U on applicability; stated as a reading limit, not a finding]

**Expanding-window (walk-forward style) folds, scikit-learn `TimeSeriesSplit`.** "returns first k folds as train set and the (k+1)th fold as test set. ... successive training sets are supersets of those that come before them. Also, it adds all surplus data to the first training partition." Folds "must represent the same duration, in order to have comparable metrics across folds." Parameters: `n_splits` (default 5), `max_train_size` (cap on training size, i.e. enables rolling instead of expanding window), `test_size` (default `n_samples // (n_splits + 1)`), `gap`. [V] https://scikit-learn.org/stable/modules/cross_validation.html ; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

**Purged K-fold and combinatorial purged CV (CPCV).** Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), ch. 7 (purged K-fold, embargo) and ch. 12 (CPCV). The book and the SSRN lecture slides (https://ssrn.com/abstract=3266136) could not be read this session (403). Descriptions of their content from implementations and secondary pages: [S]
- Purging: remove from the training set every observation whose label window overlaps in time with a label in the test set (skfolio doc, below).
- Embargo: remove from the training set observations immediately following a test block, because features carry serial correlation (skfolio doc, below).
- CPCV: purging and embargo applied over many train/test block combinations, yielding multiple backtest paths and a distribution of performance rather than one path. [S] https://www.luxalgo.com/library/concept/purged-cross-validation/

## 3. Purging and embargo: definitions and sizing parameters

Primary implementation docs (skfolio `CombinatorialPurgedCV`, which cites the book as its basis) [V]: https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html
- Purging: "removing from the training set all observations whose labels overlapped in time with those labels included in the testing set."
- Embargoing: "removing from the training set all observations that immediately follow an observation in the testing set, since financial features often incorporate series that exhibit serial correlation (like ARMA processes)."
- `purged_size`: "Number of observations to exclude from the start of each train set that are after a test set and the number of observations to exclude from the end of each training set that are before a test set."
- `embargo_size`: "Number of observations to exclude from the start of each training set that are after a test set."
- Sizes are given in number of observations (Bars), not time.

scikit-learn `TimeSeriesSplit(gap=g)` [V]: `gap` is the "Number of samples to exclude from the end of each train set before the test set." Documented example with 12 samples, `test_size=2`, `gap=2`: fold 0 trains on indices 0-3, skips 4-5, tests on 6-7. It only buffers before the test block; there is no after-test embargo because training data always precedes the test block.

What sets the size (facts from the above definitions):
- Purge width is governed by label horizon: an observation whose label reaches into the test period must be dropped (skfolio definition). For a forward label of H Bars, training observations within H Bars before the boundary have labels overlapping the test period.
- Embargo width is governed by serial correlation of features (skfolio definition). Neither skfolio nor scikit-learn prescribes a numeric rule.
- Numeric rule of thumb: the book is commonly cited as using an embargo of about 1% of Bars [U, from memory; not verified from any primary source this session]. Do not rely on it.
- Feature lookback: a feature at time t that uses a window of L past Bars means a test-region Bar at the start of the test block shares up to L Bars of raw input with the final training Bars. No primary source read this session states a purge rule for feature lookback specifically; it follows from the definition of overlap and is flagged [U] as a derived reading, not a cited rule. (The #7 charter states the project's own rule: embargo "sized to the longest feature lookback and episode length".)

## 4. Train-only scaling

- scikit-learn pitfalls guide: "The general rule is to never call `fit` on the test data." Transformations "are only learnt from the training data. For example, if you have a normalization step where you divide by the average value, the average should be the average of the train subset, not the average of all the data." Tips: split first, before any preprocessing; `fit`/`fit_transform` only on train, `transform` on both; Pipelines help prevent leakage, especially inside cross-validation. [V] https://scikit-learn.org/stable/common_pitfalls.html (section Data leakage)
- `StandardScaler`: statistics (`mean_`, `var_`, `scale_`) are computed "on the samples in the training set" and stored for later `transform`; `z = (x - u) / s`. [V] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- With walk-forward or CV folds, the same rule applies per fold: fit on that fold's train portion only. [V by the Pipeline-in-CV statement above]
- Not found in any source read: guidance on rolling or expanding-window normalisation for non-stationary series; this remains open.

## 5. Pitfalls relevant to RL on intraday forex Bars

1. Overlapping labels or horizons across the boundary (purging definition, section 3). For RL, the equivalent is an episode or reward horizon that spans the boundary; no primary source read states an RL-specific purge rule. [U]
2. Serial correlation of features across the boundary (embargo definition, section 3). Intraday Bars are highly autocorrelated in volatility and spread; stated as the motivation for embargo in the skfolio docs for financial features generally.
3. Scaler or other statistics fitted on all data (section 4).
4. Reusing the hold-out/test split for selection: Bailey et al. state hold-out is unreliable for backtests and quantify overfitting from repeated trials through PBO/CSCV. [V abstract only]
5. RL-specific generalization: Cobbe et al. (ICML 2019): "it is customary to use the same environments for both training and testing", which offers "relatively little insight into an agent's ability to generalize"; they construct separate train and test sets of environments and show agents overfit even to thousands of training levels. [V via ar5iv] https://ar5iv.labs.arxiv.org/html/1812.02341 (PMLR: https://proceedings.mlr.press/v97/cobbe19a.html). Their setting is procedurally generated games, not market data; the finding that train/test must be separated is what transfers.
6. Folds must cover the same duration for comparable metrics (scikit-learn, section 2); sessions/gaps in forex mean equal Bar counts are not equal elapsed time. [derived, U]

## 6. Gaps in this research

- Lopez de Prado book chapters 7 and 12, and the SSRN slides: not readable; purging/embargo semantics are confirmed from skfolio (an implementation citing the book), not from the book. Embargo-size numeric guidance unverified.
- Bergmeir et al. 2018 full text not read; only the abstract via search.
- No primary source found on FX-specific split practice (session boundaries, weekend Gaps, rollover) or on RL episode-length embargo.
- No primary source found on rolling/expanding normalisation for non-stationary features.
