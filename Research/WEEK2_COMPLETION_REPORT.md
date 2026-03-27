# Week 2 Completion Report — Ablation Study & Baselines

**Date:** June 2025  
**Scope:** Improvements 4, 5, 6 from RESEARCH_UPGRADE_PLAN.md  
**Status:** ✅ All 3 improvements implemented and verified  

---

## Executive Summary

Week 2 adds the experimental infrastructure that every conference paper needs:
- **"Is your method better than simple alternatives?"** → 6 academic baselines
- **"Does every component actually help?"** → 8 ablation experiments
- **"Which features matter most?"** → SHAP + permutation + group ablation

All three files are verified end-to-end on BAC/PNC with honest temporal barriers (train ≤ 2020-12-31, OOS > 2020-12-31).

---

## Improvement 4: Academic Baselines (`Research/baselines.py`)

### What Was Built

6 baseline strategies that a reviewer would expect to see in a pairs trading paper:

| # | Baseline | Description | Academic Reference |
|---|----------|-------------|--------------------|
| 1 | **Gatev Distance** | Classic 2σ distance method with formation/trading periods | Gatev et al. (2006) |
| 2 | **OLS Cointegration** | Fixed Engle-Granger hedge ratio, z-score mean reversion | Engle & Granger (1987) |
| 3 | **Kalman-Only** | Our Kalman filter hedge ratio, but regime always NORMAL | Isolates Kalman contribution |
| 4 | **Regime-Only** | Fixed OLS hedge + RF regime classification | Isolates regime contribution |
| 5 | **Buy & Hold Stocks** | Equal-weight 50/50 long both pair stocks | Passive equity benchmark |
| 6 | **SPY Buy & Hold** | Long S&P 500 ETF (market benchmark) | Standard market benchmark |

### Design Decisions

- **All baselines use identical parameters**: same `slippage_bps=5.0`, same `train_end_date`, same period tagging
- **No look-ahead**: Gatev formation window is fixed in training period; OLS hedge ratio fit only on training data
- **Multi-pair aggregation**: `run_baselines_multi_pair()` runs across N pairs and reports win rates vs each baseline

### BAC/PNC Results (Honest OOS)

| Method | OOS Return % | OOS Sharpe | Max DD % | Trades |
|--------|-------------|-----------|---------|--------|
| **Ours (Kalman+Regime)** | -6.71 | -0.114 | -33.83 | 58 |
| Gatev Distance | 41.82 | 0.669 | -12.36 | 5 |
| OLS Cointegration | 8.83 | 0.131 | -32.30 | 51 |
| Kalman-Only (No Regime) | -26.26 | -0.494 | -40.25 | 68 |
| Regime-Only (No Kalman) | 32.01 | 0.463 | -27.86 | 44 |
| B&H Stocks (50/50) | 62.15 | 0.428 | -45.90 | 0 |
| SPY Buy & Hold | 75.18 | 0.755 | -24.50 | 0 |

### Interpretation

BAC/PNC is a pair where the full system underperforms simple baselines. This is **honest reporting** — not every pair works. The value of the baseline infrastructure is that it reveals this transparently. The universe backtest (Week 1) will find pairs where our system adds genuine alpha.

Key insight: The **Regime-Only** baseline (OOS Sharpe 0.463) significantly outperforms **Kalman-Only** (-0.494), suggesting the regime classifier is the most valuable component for BAC/PNC.


### Key Functions

```
run_all_baselines(ticker_y, ticker_x, ...) → (comparison_df, all_results_dict)
run_baselines_multi_pair(pairs, ...) → aggregated comparison DataFrame
```

---

## Improvement 5: Full Ablation Study (`Research/ablation_study.py`)

### What Was Built

Systematic component removal experiment: start with the full system, remove one component at a time, measure the OOS Sharpe change.

| # | Config | What Changes |
|---|--------|-------------|
| 0 | **Full System** | Reference (all components active) |
| 1 | −MAD Outlier Detection | `KalmanNoMAD` subclass bypasses median absolute deviation check |
| 2 | −Regime Classification | All days treated as NORMAL regime |
| 3 | −Adaptive Kalman (→OLS) | Fixed OLS hedge ratio replaces Kalman |
| 4 | −Z-Score Window (40→20) | Shorter z-score lookback |
| 5 | −Stop-Loss | `stop_loss_mult=100.0` (effectively disabled) |
| 6 | −Minimum Hold Period | `min_hold_days=0` |
| 7 | −Transaction Costs | `slippage_bps=0.0` |
| 8 | −Volatile Param Adjust | Same thresholds for NORMAL and VOLATILE |

### Design Decisions

- **Custom subclass** for MAD ablation — `KalmanNoMAD` overrides the `_mad_check` method rather than using a flag, keeping the main class clean
- **OLS replacement** for Kalman ablation — uses a proper fixed hedge ratio, not a dummy
- **Ranking by impact** — components sorted by how much removing them hurts (Sharpe_Delta), so the paper can say "the most important component is X"
- **Multi-pair aggregation** — `run_ablation_multi_pair()` averages across pairs

### BAC/PNC Results (Honest OOS)

| Config | OOS Sharpe | Δ from Full | Impact |
|--------|-----------|-------------|--------|
| Full System (reference) | -0.114 | — | — |
| −Regime Classification | -0.494 | **-0.380** | 🔴 Most valuable |
| −Volatile Param Adjust | -0.494 | **-0.380** | 🔴 Tied most valuable |
| −Minimum Hold Period | -0.418 | **-0.305** | 🔴 Important |
| −Stop-Loss | -0.174 | -0.060 | ⚪ Minor |
| −Z-Score Window | -0.139 | -0.025 | ⚪ Negligible |
| −Transaction Costs | -0.018 | +0.096 | 🟢 Expected (costs hurt) |
| −MAD Outlier Detection | +0.182 | +0.295 | 🟢 Removing helps for BAC/PNC |
| −Adaptive Kalman (→OLS) | +0.618 | **+0.732** | 🟢 OLS outperforms Kalman here |

### Interpretation

For BAC/PNC specifically:
- **Regime classification** is the most valuable component (removing it hurts Sharpe by 0.38)
- **Minimum hold period** prevents overtrading (removing it drops Sharpe by 0.30)
- **Kalman filter** actually hurts on this particular pair — the OLS fixed hedge ratio is more stable
- This is a per-pair result. The multi-pair runner will show which components are consistently valuable across many pairs

### Key Functions

```
run_ablation_single_pair(ticker_y, ticker_x, ...) → DataFrame with Sharpe_Delta
run_ablation_multi_pair(pairs, ...) → aggregated ranking by importance
```

---

## Improvement 6: Feature Importance Analysis (`Research/feature_analysis.py`)

### What Was Built

Four complementary feature importance methods for the Random Forest regime classifier:

| # | Method | Type | Reference |
|---|--------|------|-----------|
| 1 | **SHAP TreeExplainer** | Exact Shapley values for trees | Lundberg & Lee (2017) |
| 2 | **Gini/MDI Importance** | Built-in sklearn feature importance | Breiman (2001) |
| 3 | **Permutation Importance** | Model-agnostic, 10 repeats | Breiman (2001), Altmann et al. (2010) |
| 4 | **Feature Group Ablation** | Remove one group, retrain | Our contribution |

Plus a **consensus ranking** that averages ranks across all three individual methods.

### Feature Groups Defined

| Group | Features | Purpose |
|-------|----------|---------|
| Volatility | vol_5, vol_10, vol_20, vol_40, vol_60 | Multi-scale realized volatility |
| Vol Dynamics | vol_trend, vol_accel | Volatility momentum and acceleration |
| Drawdowns | dd_10, dd_20, dd_40, dd_60 | Multi-scale drawdowns |
| Returns | ret_5, ret_10, ret_20 | Cumulative returns at various horizons |
| Trend | trend, trend_strength | Price trend indicators |
| Range | range_expansion | Bollinger band width expansion |
| Pair Corr | pair_corr, corr_change | Rolling pair correlation and change |
| Pair Vol | pair_vol, pair_vol_spike | Pair spread volatility |

### BAC/PNC Results

#### Consensus Ranking (Top 10)

| Rank | Feature | SHAP Rank | Gini Rank | Perm Rank | Average |
|------|---------|-----------|-----------|-----------|---------|
| 1 | **vol_20** | 1 | 1 | 1 | **1.0** |
| 2 | pair_vol | 6 | 9 | 3 | 6.0 |
| 3 | pair_vol_spike | 9 | 10 | 2 | 7.0 |
| 4 | range_expansion | 3 | 3 | 17 | 7.7 |
| 5 | vol_10 | 2 | 2 | 21 | 8.3 |
| 6 | dd_60 | 4 | 4 | 20 | 9.3 |
| 7 | vol_60 | 8 | 6 | 15 | 9.7 |
| 8 | vol_40 | 5 | 5 | 19 | 9.7 |
| 9 | vol_accel | 12 | 13 | 4 | 9.7 |
| 10 | vol_5 | 7 | 8 | 16 | 10.3 |

#### Feature Group Ablation

| Group Removed | Accuracy | Δ from Baseline | F1 | Impact |
|---------------|----------|-----------------|-----|--------|
| All Features (baseline) | 0.894 | — | 0.897 | — |
| **−Volatility** | 0.774 | **-0.121** | 0.789 | 🔴 Critical |
| −Pair Vol | 0.861 | -0.034 | 0.865 | 🔴 Important |
| −Vol Dynamics | 0.877 | -0.018 | 0.880 | 🔴 Moderate |
| −Trend | 0.889 | -0.005 | 0.892 | ⚪ Minor |
| −Range | 0.892 | -0.003 | 0.896 | ⚪ Minor |
| −Pair Corr | 0.892 | -0.003 | 0.895 | ⚪ Minor |
| −Returns | 0.893 | -0.002 | 0.897 | ⚪ Negligible |
| −Drawdowns | 0.920 | +0.026 | 0.922 | 🟢 Removing helps |

### Interpretation

1. **`vol_20` is the undisputed #1 feature** — ranked #1 by all three methods. 20-day realized volatility is the strongest regime signal.
2. **Volatility features dominate** — 6 of the top 10 consensus features are volatility measures. Removing the entire Volatility group drops classifier accuracy by 12.1%.
3. **Pair-specific features (pair_vol, pair_vol_spike)** are more important than expected — they're #2 and #3 in consensus.
4. **Drawdowns can be dropped** without hurting (and may slightly improve) classifier accuracy — candidate for feature pruning.
5. **SHAP vs Permutation disagreements** (e.g., range_expansion: SHAP_rk=3 vs Perm_rk=17) suggest feature correlations — SHAP accounts for interactions, permutation doesn't.

### Key Functions

```
run_feature_analysis(ticker_y, ticker_x, ...) → dict with 'shap', 'gini', 'permutation', 'group_ablation', 'consensus'
run_feature_analysis_multi_pair(pairs, ...) → global feature ranking across all pairs
```

---

## Files Created/Modified

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `Research/baselines.py` | ~380 | ✅ New | 6 academic baseline strategies |
| `Research/ablation_study.py` | ~370 | ✅ New | 8 ablation configurations + runners |
| `Research/feature_analysis.py` | ~410 | ✅ New | SHAP + Gini + permutation + group ablation |
| `Research/WEEK2_COMPLETION_REPORT.md` | — | ✅ New | This report |

## Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| `shap` | 0.51.0 | SHAP TreeExplainer for Random Forest |

All other dependencies (scikit-learn, statsmodels, scipy, matplotlib) were already installed from Week 1.

---

## Verification Summary

All 3 files were verified with:

1. **Import test** — All classes, functions, and constants import without errors
2. **End-to-end BAC/PNC run** — Each file executes completely with real data
3. **Output validation** — Correct column names, shapes, and value ranges

```
✅ baselines.py     — 7 strategies compared, OOS comparison table generated
✅ ablation_study.py — 9 configs (ref + 8 ablations), Sharpe_Delta computed
✅ feature_analysis.py — 5 analysis outputs (shap, gini, permutation, group_ablation, consensus)
```

---

## What Week 2 Enables for the Paper

With Weeks 1 + 2 combined, you can now write:

1. **Section 4.1: Baseline Comparison** — "Our method vs 6 alternatives across N pairs"
2. **Section 4.2: Ablation Study** — "Table showing marginal value of each component"
3. **Section 4.3: Feature Importance** — "SHAP analysis reveals vol_20 as the dominant regime signal"
4. **Figure 3** — Consensus feature ranking bar chart
5. **Figure 4** — Component importance ranked by ablation Sharpe delta

These sections directly address the most common reviewer questions:
- *"How does this compare to simpler methods?"* → Baselines table
- *"Is every component necessary?"* → Ablation table
- *"What drives the model's decisions?"* → SHAP + consensus ranking

---

## Next Steps (Week 3: Statistical Rigor)

Week 3 will add the statistical tests that make results publishable:

| # | Improvement | Description |
|---|-------------|-------------|
| 7 | Paired t-tests / Wilcoxon | Statistical comparison vs baselines |
| 8 | Multiple hypothesis correction | Benjamini-Hochberg across pairs and baselines |
| 9 | Confidence intervals | Bootstrap CIs for all reported metrics |
| 10 | Effect size reporting | Cohen's d for practical significance |
