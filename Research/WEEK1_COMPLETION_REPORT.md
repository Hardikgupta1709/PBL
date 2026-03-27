# Week 1 Completion Report — Experimental Foundation

**Period:** March 10–16, 2026  
**Theme:** Fix the science before writing about it  
**Status:** ✅ COMPLETE  

---

## Executive Summary

Week 1 rebuilt the experimental foundation of the pairs trading research pipeline. Three critical improvements were implemented: (1) strict temporal train/validate/test protocol eliminating look-ahead bias, (2) a statistically correct Monte Carlo significance test, and (3) a universe-scale backtest infrastructure covering 50+ pairs across 10 sectors. These changes transform the project from an engineering demo into a defensible scientific experiment.

---

## Improvement 1: Strict Train/Validate/Test Split

**File:** `Core_Strategy/conservative_strategy.py` (714 lines, complete rewrite)  
**Problem:** The old code trained the Random Forest regime classifier on data that overlapped with the signal evaluation window, introducing look-ahead bias. This made backtest results unrealistically optimistic.  

### What Changed

| Aspect | Before (v1) | After (v2) |
|--------|-------------|------------|
| RF training window | `train_period` integer (rolling) | `train_end_date` string (strict cutoff: 2020-12-31) |
| Look-ahead bias | Present — RF saw future data | Eliminated — RF frozen at train_end_date |
| Transaction costs | Applied post-hoc | Integrated inside backtest loop |
| Period tagging | None | Every row tagged `'train'` or `'oos'` |
| Performance reporting | Single aggregate | Per-period via `get_performance_metrics(period=...)` |
| Baseline comparison | None | No-regime baseline computed alongside |
| Code style | `print()` statements | Python `logging` module |
| Type hints | Absent | Full type annotations on all public methods |
| Docstrings | Minimal | NumPy-style docstrings on every class/function |

### Key Design Decisions

1. **Temporal barrier enforcement:** `run_backtest()` takes `train_end_date` as a string parameter. The RF classifier calls `.train()` only on rows where `index <= train_end_date`. This is not a convention — it is enforced in code with a `train_mask`.

2. **Transaction cost integration:** Signal changes (`data['final_signal'].diff().abs()`) are detected and a per-change cost of `slippage_bps / 10,000` is deducted from daily returns. This happens *inside* the backtest loop, not as a post-hoc adjustment.

3. **Period column:** Every row in the results DataFrame gets `period='train'` or `period='oos'`. Downstream analysis (validator, universe backtest) filters on this column.

### New Components Added

- **`calculate_half_life(spread)`** — OLS-based half-life estimation for mean reversion speed
- **`calculate_zscore(spread, window)`** — Rolling z-score with expanding window fallback
- **`StrictRegimeClassifier.predict_proba()`** — Returns class probabilities for downstream analysis
- **`ConservativeSystem.get_trade_analysis(period=...)`** — Extracts individual trade records with entry/exit dates, z-scores, regime, duration, PnL
- **`ConservativeSystem.slice_period(start, end)`** — Custom date range slicing

### Honest Results (BAC/PNC)

```
OOS Period (2021-01-01 → 2025-06-27):
  Strategy:       -6.71% total | -1.54% annual | Sharpe -0.114
  No-Regime Base: -26.26% total | Sharpe -0.494
  Market (SPY):   +75.18% total | Sharpe +0.755
  
  OOS Trades: 58 | Win Rate: 41.4% | CRISIS: 1.2%
```

**Interpretation:** With honest temporal barriers, BAC/PNC does not beat the market. However, the regime filter adds significant value vs the no-regime baseline (+19.55% improvement). This is the honest truth — the universe backtest will identify which pairs actually work OOS.

---

## Improvement 2: Fixed Monte Carlo + Full Validation Suite

**File:** `Core_Strategy/strategy_validator.py` (620+ lines, complete rewrite)  
**Problem:** The old Monte Carlo test shuffled trade returns — which preserves the mean and thus always produces similar results. It tested the *ordering* of trades, not whether entry *timing* has skill.

### What Changed

| Component | Before (v1) | After (v2) |
|-----------|-------------|------------|
| Monte Carlo method | Shuffle trade PnLs | Random entry/exit dates on same spread |
| Simulations | 1,000 | 10,000 (configurable) |
| Walk-forward | Sliding window, 4–5 folds | Expanding window, 8–10+ folds, 63-day steps |
| Bootstrap CI | Not present | 10,000 resamples for Sharpe ratio |
| Multiple testing | Not present | Benjamini-Hochberg FDR correction |
| Stress testing | 3 crisis periods | 6 crisis periods (incl. 2023 Banking Crisis) |
| Sensitivity grid | 3×3 = 9 combos | 7×5×3 = 105 combos (entry × exit × window) |
| Statistical tests | JB + t-test | + ADF stationarity, Ljung-Box autocorrelation, skewness/kurtosis |
| Config | Hardcoded | Reads from `Research/config_experiments.yaml` |

### Monte Carlo Fix (Critical)

**Old (WRONG):** Shuffled trade PnLs → null distribution ≈ same mean → p-values meaningless  
**New (CORRECT):** 
1. Take the actual spread return series from the backtest
2. Generate 10,000 random strategies: pick random entry dates, hold for random durations drawn from the empirical distribution, random direction (long/short)
3. Compute Sharpe for each random strategy
4. p-value = P(random_Sharpe ≥ actual_Sharpe)

This tests **whether the timing of entries matters**, which is the fundamental question for a mean-reversion strategy.

### New Classes

- **`WalkForwardAnalyzer`** — Expanding-window protocol with quarterly 63-day steps
- **`MonteCarloValidator`** — Random entry/exit date null hypothesis
- **`BootstrapAnalyzer`** — Non-parametric bootstrap for Sharpe CI
- **`StressTester`** — 6 named crisis periods with strategy vs market alpha
- **`SensitivityAnalyzer`** — Full parameter grid with OOS-only evaluation
- **`StatisticalTester`** — ADF, t-test, Ljung-Box, JB, skewness, kurtosis
- **`benjamini_hochberg()`** — FDR correction for testing many pairs
- **`StrategyValidator`** — Unified runner: calls all 6 tests, computes weighted score, generates text report

### Honest Results (BAC/PNC)

```
Monte Carlo (1,000 quick sims):
  Actual Sharpe: -0.114
  p-value (Sharpe): 0.544 → ❌ FAIL
  
Bootstrap CI:
  Sharpe: -0.047 [-0.937, 0.884] → ❌ FAIL (CI includes zero)
```

---

## Improvement 3: Universe Backtest (50+ Pairs)

**File:** `Research/universe_backtest.py` (420+ lines, new file)  
**Problem:** Testing on 2–3 cherry-picked pairs proves nothing about the method's generalizability.

### What It Does

1. **Pair Discovery:** Scans all 10 sectors in `STOCK_UNIVERSE` (74 stocks), scores every intra-sector pair on cointegration (50pts), correlation (20pts), ADF stationarity (30pts), and half-life bonus
2. **Filtering:** Keeps pairs scoring ≥ 40 (configurable), up to 10 per sector
3. **Full Backtest:** Runs `ConservativeSystem.run_backtest()` on each pair with strict temporal split
4. **Per-Pair Metrics:** OOS Sharpe, return, alpha vs SPY, trades, win rate, max drawdown
5. **Monte Carlo p-value:** Quick MC (1,000 sims) per pair
6. **Bootstrap CI:** Per-pair Sharpe confidence interval
7. **BH Correction:** Benjamini-Hochberg across all pair-level MC p-values
8. **Aggregate Analysis:** Median Sharpe, hit rate, sector breakdown, train→OOS overfit ratio
9. **Reports:** CSV (`universe_summary.csv`) + text report (`universe_report.txt`)

### Pipeline

```
discover_pairs() → backtest_universe() → aggregate_analysis() → generate_universe_report()
```

### Key Design Features

- Rate-limited API calls (0.5s delay between pairs)
- Graceful error handling per pair (one failure doesn't stop the pipeline)
- Train→OOS overfit ratio to detect systematic overfitting
- Sector breakdown to identify which sectors work best

---

## Supporting Infrastructure Created

| File | Purpose |
|------|---------|
| `Research/__init__.py` | Package marker |
| `Research/config_experiments.yaml` | All experiment hyperparameters centralized |
| `Research/results/` | Output directory for CSVs and reports |

### Config Structure (`config_experiments.yaml`)

```yaml
DATA_START: "2015-01-01"
TRAIN_END: "2020-12-31"
VALIDATE_START: "2021-01-01"
TEST_END: "2025-06-30"

strategy: {entry_z_normal: 1.5, exit_z_normal: 0.5, ...}
regime: {n_estimators: 300, max_depth: 6, ...}
kalman: {delta: 1e-4, var_e: 1e-3, ...}
monte_carlo: {n_simulations: 10000, ...}
walk_forward: {step_days: 63, min_folds: 8, ...}
bootstrap: {n_resamples: 10000, ...}
sensitivity: {entry_z_values: [1.0, 1.25, ..., 3.0], ...}
```

---

## Files Modified / Created

| Action | File | Lines |
|--------|------|-------|
| ✅ Rewritten | `Core_Strategy/conservative_strategy.py` | 714 |
| ✅ Rewritten | `Core_Strategy/strategy_validator.py` | 620+ |
| ✅ Created | `Research/universe_backtest.py` | 420+ |
| ✅ Created | `Research/config_experiments.yaml` | 95 |
| ✅ Created | `Research/__init__.py` | 5 |
| 📦 Archived | `Archive/old_code/conservative_strategy_v1.py` | 582 |
| 📦 Archived | `Archive/old_code/strategy_validator_v1.py` | 673 |

---

## Verification

All three files import and pass smoke tests:
```
✅ conservative_strategy imports OK
✅ strategy_validator imports OK
✅ universe_backtest imports OK
✅ Kalman update: innovation=0.0000, hedge=2.0000
✅ BH correction: 1/4 significant
✅ Signal generation: 36 trading days out of 100
All Week 1 smoke tests PASSED
```

End-to-end backtest on BAC/PNC (2015-2025, train ≤ 2020) completed successfully with 58 OOS trades, proper period tagging, and integrated transaction costs.

---

## Score Impact

| Dimension | Before Week 1 | After Week 1 |
|-----------|---------------|--------------|
| Methodology Rigor | 3/10 | 5/10 |
| Experimental Design | 3/10 | 5/10 |
| Statistical Soundness | 2/10 | 4/10 |
| Reproducibility | 4/10 | 5/10 |
| **Overall Estimate** | **3/10** | **4.5/10** |

**Next:** Week 2 — Ablation Study & Baselines (proving each component earns its place)
