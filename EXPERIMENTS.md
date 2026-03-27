# EXPERIMENTS.md — Per-Experiment Reproduction Guide

This document describes how to reproduce each experiment in the paper
individually. For a one-command full reproduction, use:

```bash
python Research/reproduce.py
```

All experiments use `random_seed=42` and the temporal split
`train ≤ 2020-12-31 | OOS > 2020-12-31`.

---

## Table of Contents

1. [Baseline Comparison](#1-baseline-comparison)
2. [Ablation Study](#2-ablation-study)
3. [Feature Importance Analysis](#3-feature-importance-analysis)
4. [Statistical Tests](#4-statistical-tests)
5. [Walk-Forward Analysis](#5-walk-forward-analysis)
6. [Cointegration Stability](#6-cointegration-stability)
7. [Regime Detection Evaluation](#7-regime-detection-evaluation)
8. [Performance Attribution](#8-performance-attribution)
9. [Figures](#9-figures)
10. [Tables](#10-tables)

---

## 1. Baseline Comparison

**Paper reference:** Table 2, Figure 4  
**Module:** `Research/baselines.py`  
**Function:** `run_all_baselines(ticker_y, ticker_x, ...)`

Compares 7 strategies on each pair:
- Simple Z-score (fixed OLS hedge, Bollinger-style entry)
- Johansen-based (eigenvector hedge ratio from Johansen test)
- Distance method (minimum SSD, normalised price distance)
- Copula-based (Clayton copula, conditional probability signals)
- Ornstein-Uhlenbeck (MLE OU parameters, half-life signals)
- Machine-learning (gradient-boosted classifier predicting spread direction)
- **Our method** (adaptive Kalman + regime gating)

### Standalone reproduction

```python
from Research.baselines import run_all_baselines

comp_df, details = run_all_baselines(
    "BAC", "PNC",
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31", slippage_bps=5.0,
)
print(comp_df[["Strategy", "OOS_Sharpe", "OOS_Return", "Max_DD"]])
```

### Expected output

| Strategy | OOS Sharpe | OOS Return | Max DD |
|----------|-----------|-----------|--------|
| Our Method | −0.126 | −4.2% | −12.3% |
| Simple Z-score | −0.31 | −8.1% | −15.2% |
| Johansen | −0.18 | −5.6% | −11.8% |
| Distance | −0.42 | −10.3% | −16.7% |
| Copula | −0.25 | −6.9% | −13.4% |
| OU | −0.19 | −5.8% | −12.1% |
| ML | −0.33 | −8.7% | −14.9% |

*(Values are approximate; slight shifts expected with yfinance updates.)*

---

## 2. Ablation Study

**Paper reference:** Table 3, Figure 3  
**Module:** `Research/ablation_study.py`  
**Function:** `run_ablation_single_pair(ticker_y, ticker_x, ...)`

Tests 9 configurations:
1. Full system (Kalman + regime + all features)
2. No regime filter
3. No Kalman (fixed OLS hedge)
4. No adaptive bands
5. No volume features
6. No macro features
7. Static hedge (no filter)
8. Simple Z-score only
9. Random entry baseline

### Standalone reproduction

```python
from Research.ablation_study import run_ablation_single_pair

abl_df = run_ablation_single_pair(
    "BAC", "PNC",
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31", slippage_bps=5.0,
)
print(abl_df[["Config", "OOS_Sharpe", "OOS_Return"]])
```

---

## 3. Feature Importance Analysis

**Paper reference:** Table 5, Figure 7  
**Module:** `Research/feature_analysis.py`  
**Function:** `run_feature_analysis(ticker_y, ticker_x, ...)`

Three importance methods:
- **SHAP values** (TreeExplainer on the regime classifier)
- **Gini importance** (built-in RF feature importances)
- **Permutation importance** (OOS accuracy drop per feature)

Produces a consensus ranking via rank-averaging.

### Standalone reproduction

```python
from Research.feature_analysis import run_feature_analysis

result = run_feature_analysis(
    "BAC", "PNC",
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31",
)
print(result["consensus"].head(10))
```

---

## 4. Statistical Tests

**Paper reference:** Table 1 confidence intervals, Section IV-D  
**Module:** `Research/statistical_tests.py`  
**Functions:** `run_full_statistical_analysis()`, `run_statistical_analysis_multi_pair()`

Tests performed:
- **Bootstrap CIs** (10,000 resamples for Sharpe, return, max DD)
- **Monte Carlo significance** (10,000 random-entry strategies)
- **Benjamini-Hochberg correction** (multi-pair FDR at α=0.05)

### Standalone reproduction

```python
from Research.statistical_tests import run_statistical_analysis_multi_pair

result = run_statistical_analysis_multi_pair(
    [("BAC", "PNC"), ("WFC", "MS"), ("CVX", "OXY")],
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31", n_bootstrap=10000,
)
```

---

## 5. Walk-Forward Analysis

**Paper reference:** Table 4, Figure 10  
**Module:** `Core_Strategy/strategy_validator.py`  
**Class:** `WalkForwardAnalyzer`

Configuration:
- Minimum training window: 504 days (~2 years)
- Step size: 63 days (~1 quarter)
- Test window: 126 days (~6 months)
- Minimum folds: 8

### Standalone reproduction

```python
import yfinance as yf
from Core_Strategy.strategy_validator import WalkForwardAnalyzer

data = yf.download(["BAC", "PNC", "SPY"],
                   start="2015-01-01", end="2025-06-30",
                   auto_adjust=True)["Close"]
wfa = WalkForwardAnalyzer(min_train_days=504, step_days=63,
                          test_days=126, min_folds=8)
wf_df = wfa.run(data["BAC"], data["PNC"], data["SPY"], slippage_bps=5.0)
print(wf_df[["fold", "test_sharpe", "test_return"]])
```

---

## 6. Cointegration Stability

**Paper reference:** Figure 6  
**Module:** `Research/cointegration_analysis.py`  
**Function:** `run_cointegration_multi_pair()`

Rolling window analysis:
- Window: 126 trading days (~6 months)
- Tests: ADF statistic + p-value, Hurst exponent
- Tracks fraction of windows where cointegration holds at 5%

### Standalone reproduction

```python
from Research.cointegration_analysis import run_cointegration_multi_pair

summary = run_cointegration_multi_pair(
    [("BAC", "PNC"), ("WFC", "MS"), ("CVX", "OXY")],
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31", window=126,
)
print(summary)
```

---

## 7. Regime Detection Evaluation

**Paper reference:** Table 7  
**Module:** `Research/regime_evaluation.py`  
**Function:** `run_regime_evaluation(ticker_y, ticker_x, ...)`

Compares three classifiers:
- **Random Forest** (our method, 12 macro-financial features)
- **HMM** (2-state Hidden Markov Model via hmmlearn, optional)
- **Simple rules** (VIX > 25 = crisis, trailing vol threshold)

### Standalone reproduction

```python
from Research.regime_evaluation import run_regime_evaluation

result = run_regime_evaluation(
    "BAC", "PNC",
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31", slippage_bps=5.0,
)
print(result["summary"])
```

---

## 8. Performance Attribution

**Paper reference:** Table 8, Figure 9  
**Module:** `Research/attribution.py`  
**Function:** `run_attribution_multi_pair(pairs, ...)`

Decomposes OOS returns into 4 alpha sources:
1. **Spread mean-reversion** (base cointegration signal)
2. **Kalman adaptation** (dynamic hedge ratio updates)
3. **Regime gating** (trade suppression in non-MR regimes)
4. **Band calibration** (adaptive entry/exit thresholds)

### Standalone reproduction

```python
from Research.attribution import run_attribution_multi_pair

attr_df = run_attribution_multi_pair(
    [("BAC", "PNC"), ("WFC", "MS"), ("CVX", "OXY")],
    start_date="2015-01-01", end_date="2025-06-30",
    train_end_date="2020-12-31", slippage_bps=5.0,
)
print(attr_df)
```

---

## 9. Figures

All 11 figures are generated by `Research/generate_paper_figures.py`:

| Figure | Description | Function |
|--------|-------------|----------|
| 1 | System architecture diagram | `fig1_architecture()` |
| 2 | Spread + regime bands + signals | `fig2_regime_bands()` |
| 3 | Ablation waterfall chart | `fig3_ablation()` |
| 4 | Baseline comparison bar chart | `fig4_baselines()` |
| 5 | Entry threshold sensitivity heatmap | `fig5_sensitivity()` |
| 6 | Rolling cointegration time series | `fig6_cointegration()` |
| 7 | SHAP feature importance beeswarm | `fig7_feature_importance()` |
| 8 | Cumulative OOS returns (3 pairs) | `fig8_cumulative_returns()` |
| 9 | Attribution stacked decomposition | `fig9_attribution()` |
| 10 | Walk-forward fold Sharpes | `fig10_walkforward()` |
| 11 | Multi-pair comparison (bonus) | `fig_bonus_multipair()` |

### Standalone reproduction

```bash
python Research/generate_paper_figures.py
```

Or individually:

```python
from Research.generate_paper_figures import fig3_ablation
fig3_ablation()  # Saves to Paper/figures/
```

---

## 10. Tables

All 8 tables are generated by `Research/generate_paper_tables.py`:

| Table | Description | File |
|-------|-------------|------|
| 1 | Pair summary statistics | `table1_pair_summary.tex` |
| 2 | Baseline comparison | `table2_baselines.tex` |
| 3 | Ablation results | `table3_ablation.tex` |
| 4 | Walk-forward fold results | `table4_walkforward.tex` |
| 5 | Feature importance ranking | `table5_features.tex` |
| 6 | Paper trading performance | `table6_paper_trading.tex` |
| 7 | Regime method comparison | `table7_regime_comparison.tex` |
| 8 | Attribution decomposition | `table8_attribution.tex` |

### Standalone reproduction

```bash
python Research/generate_paper_tables.py
```

---

## Environment Notes

- **Python version:** 3.12.9 (tested)
- **OS:** macOS / Linux (Docker image uses Debian slim)
- **hmmlearn:** Optional. If not installed, HMM regime comparison is skipped.
- **Data source:** Yahoo Finance via `yfinance`. Data may shift slightly
  over time as Yahoo adjusts historical prices.
- **Runtime:** Full reproduction takes ~30-45 minutes. The `--quick` flag
  reduces Monte Carlo from 10,000 to 1,000 iterations (~5-10 min).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: hmmlearn` | Optional; HMM comparison is skipped |
| `yfinance` download fails | Check internet; retry after 60s (rate limit) |
| Plots don't display | Set `MPLBACKEND=Agg` (headless) |
| Slight numerical differences | Expected due to yfinance data updates |
| Docker build slow | First build compiles numpy; subsequent builds use cache |
