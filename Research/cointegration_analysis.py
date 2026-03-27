"""
Cointegration Stability Analysis
==================================
Week 3, Improvement 9: "Pairs that were cointegrated may break."

Provides rolling analysis of pair relationship health:

    1. Rolling Cointegration Tests — Engle-Granger p-value over time (126-day window)
    2. Rolling Hurst Exponent — Mean-reversion strength over time
    3. Rolling Half-Life — Speed of mean reversion over time
    4. PnL × Stability Correlation — Does profitability track pair health?

Output:
    - Time-series DataFrame with all rolling metrics
    - Summary statistics (% of time cointegrated, avg Hurst, etc.)
    - Correlation table: rolling_coint_p vs rolling_pnl

References:
    Engle & Granger (1987) "Co-Integration and Error Correction"
    Hurst (1951) — H < 0.5 → mean-reverting; H > 0.5 → trending
    Vidyamurthy (2004) "Pairs Trading" — half-life application
"""

import numpy as np
import pandas as pd
import os
import sys
import logging
import warnings
from typing import Dict, Optional

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, ConservativeKalman, download_data,
    calculate_zscore, calculate_half_life
)
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# =============================================================================
# 1. ROLLING COINTEGRATION TEST
# =============================================================================

def rolling_cointegration(stock_y: pd.Series, stock_x: pd.Series,
                          window: int = 126, step: int = 5) -> pd.DataFrame:
    """
    Rolling Engle-Granger cointegration test.

    Parameters
    ----------
    stock_y, stock_x : pd.Series
        Price series for the pair.
    window : int
        Rolling window size in trading days (126 ≈ 6 months).
    step : int
        Step size to reduce computation (test every `step` days).

    Returns
    -------
    DataFrame with columns: date, coint_pvalue, coint_stat, is_cointegrated
    """
    data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()
    n = len(data)
    rows = []

    for i in range(window, n, step):
        window_data = data.iloc[i - window:i]
        date = data.index[i]

        try:
            stat, pval, _ = coint(window_data['Y'].values,
                                   window_data['X'].values)
            rows.append({
                'date': date,
                'coint_pvalue': pval,
                'coint_stat': stat,
                'is_cointegrated': pval < 0.05,
            })
        except Exception:
            rows.append({
                'date': date,
                'coint_pvalue': np.nan,
                'coint_stat': np.nan,
                'is_cointegrated': False,
            })

    return pd.DataFrame(rows).set_index('date')


# =============================================================================
# 2. ROLLING HURST EXPONENT
# =============================================================================

def hurst_exponent(series: np.ndarray) -> float:
    """
    Estimate the Hurst exponent using the rescaled range (R/S) method.

    H < 0.5 → mean-reverting (good for pairs trading)
    H = 0.5 → random walk
    H > 0.5 → trending

    Uses multiple sub-window sizes for robust estimation.
    """
    n = len(series)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 512)
    # Use a range of window sizes
    sizes = []
    k = 8
    while k <= max_k:
        sizes.append(k)
        k = int(k * 1.5)

    if len(sizes) < 3:
        return 0.5

    rs_values = []
    for size in sizes:
        n_windows = n // size
        if n_windows < 1:
            continue

        rs_list = []
        for j in range(n_windows):
            window = series[j * size:(j + 1) * size]
            mean = window.mean()
            deviations = window - mean
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = window.std(ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((np.log(size), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5

    log_sizes = np.array([x[0] for x in rs_values])
    log_rs = np.array([x[1] for x in rs_values])

    # Linear regression: log(R/S) = H * log(n) + c
    model = LinearRegression()
    model.fit(log_sizes.reshape(-1, 1), log_rs)
    H = float(np.clip(model.coef_[0], 0.0, 1.0))

    return H


def rolling_hurst(spread: pd.Series, window: int = 126,
                  step: int = 5) -> pd.DataFrame:
    """
    Rolling Hurst exponent on the spread series.

    Parameters
    ----------
    spread : pd.Series
        The spread (or log-spread) between the two assets.
    window : int
        Window size for Hurst calculation.
    step : int
        Step size.

    Returns
    -------
    DataFrame with columns: date, hurst, is_mean_reverting
    """
    n = len(spread)
    rows = []

    for i in range(window, n, step):
        date = spread.index[i]
        window_data = spread.iloc[i - window:i].values

        H = hurst_exponent(window_data)
        rows.append({
            'date': date,
            'hurst': H,
            'is_mean_reverting': H < 0.5,
        })

    return pd.DataFrame(rows).set_index('date')


# =============================================================================
# 3. ROLLING HALF-LIFE
# =============================================================================

def rolling_half_life(spread: pd.Series, window: int = 126,
                      step: int = 5) -> pd.DataFrame:
    """
    Rolling half-life of mean reversion estimated via OLS on spread lag.

    Half-life ∈ [5, 60] days clipped. Lower = faster mean reversion = better.
    """
    n = len(spread)
    rows = []

    for i in range(window, n, step):
        date = spread.index[i]
        window_spread = spread.iloc[i - window:i]

        hl = calculate_half_life(window_spread)
        rows.append({
            'date': date,
            'half_life': hl,
            'fast_reversion': hl < 20,
        })

    return pd.DataFrame(rows).set_index('date')


# =============================================================================
# 4. ROLLING HEDGE RATIO STABILITY
# =============================================================================

def rolling_hedge_ratio(stock_y: pd.Series, stock_x: pd.Series,
                        window: int = 126, step: int = 5) -> pd.DataFrame:
    """Rolling OLS hedge ratio to track structural stability."""
    data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()
    n = len(data)
    rows = []

    for i in range(window, n, step):
        date = data.index[i]
        w = data.iloc[i - window:i]

        model = LinearRegression()
        model.fit(w['X'].values.reshape(-1, 1), w['Y'].values)
        hr = model.coef_[0]
        r2 = model.score(w['X'].values.reshape(-1, 1), w['Y'].values)

        rows.append({
            'date': date,
            'hedge_ratio': hr,
            'r_squared': r2,
        })

    return pd.DataFrame(rows).set_index('date')


# =============================================================================
# 5. ROLLING ADF TEST ON SPREAD
# =============================================================================

def rolling_adf(spread: pd.Series, window: int = 126,
                step: int = 5) -> pd.DataFrame:
    """
    Rolling Augmented Dickey-Fuller test on the spread.
    p < 0.05 → spread is stationary (good).
    """
    n = len(spread)
    rows = []

    for i in range(window, n, step):
        date = spread.index[i]
        window_data = spread.iloc[i - window:i].values

        try:
            stat, pval, *_ = adfuller(window_data, maxlag=15)
            rows.append({
                'date': date,
                'adf_stat': stat,
                'adf_pvalue': pval,
                'is_stationary': pval < 0.05,
            })
        except Exception:
            rows.append({
                'date': date,
                'adf_stat': np.nan,
                'adf_pvalue': np.nan,
                'is_stationary': False,
            })

    return pd.DataFrame(rows).set_index('date')


# =============================================================================
# 6. UNIFIED ANALYSIS
# =============================================================================

def run_cointegration_analysis(
    ticker_y: str = 'BAC', ticker_x: str = 'PNC',
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    window: int = 126,
    step: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run full cointegration stability analysis for a single pair.

    Returns dict with:
        - rolling_metrics: merged DataFrame of all rolling analyses
        - summary: summary statistics
        - pnl_stability_corr: correlation between pair health and PnL
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  COINTEGRATION STABILITY ANALYSIS: {ticker_y}/{ticker_x}")
        print(f"  Window: {window}d | Step: {step}d")
        print(f"{'=' * 70}")

    # Download data
    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    # Run our backtest to get spread and PnL
    system = ConservativeSystem()
    data = system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        slippage_bps=5.0,
        verbose=False,
    )

    spread = data['spread']
    train_end = pd.Timestamp(train_end_date)

    # --- 1. Rolling Cointegration ---
    if verbose:
        print(f"\n  1. Rolling cointegration test (Engle-Granger)...")
    coint_df = rolling_cointegration(stock_y, stock_x, window=window, step=step)

    # --- 2. Rolling Hurst ---
    if verbose:
        print(f"  2. Rolling Hurst exponent...")
    hurst_df = rolling_hurst(spread, window=window, step=step)

    # --- 3. Rolling Half-Life ---
    if verbose:
        print(f"  3. Rolling half-life...")
    hl_df = rolling_half_life(spread, window=window, step=step)

    # --- 4. Rolling Hedge Ratio ---
    if verbose:
        print(f"  4. Rolling hedge ratio stability...")
    hr_df = rolling_hedge_ratio(stock_y, stock_x, window=window, step=step)

    # --- 5. Rolling ADF ---
    if verbose:
        print(f"  5. Rolling ADF test on spread...")
    adf_df = rolling_adf(spread, window=window, step=step)

    # --- 6. Merge all rolling metrics ---
    merged = coint_df.join(hurst_df, how='outer')
    merged = merged.join(hl_df, how='outer')
    merged = merged.join(hr_df, how='outer')
    merged = merged.join(adf_df, how='outer')
    merged = merged.ffill()

    # Add period labels
    merged['period'] = 'train'
    merged.loc[merged.index > train_end, 'period'] = 'oos'

    # --- 7. Rolling PnL (to correlate with stability) ---
    rolling_pnl = data['strategy_return'].rolling(window).sum()
    # Align to merged index
    merged['rolling_pnl'] = rolling_pnl.reindex(merged.index, method='ffill')

    # --- 8. Summary statistics ---
    oos_merged = merged[merged['period'] == 'oos'].dropna(subset=['coint_pvalue'])

    summary = {}
    if len(oos_merged) > 0:
        summary['oos_pct_cointegrated'] = oos_merged['is_cointegrated'].mean() * 100
        summary['oos_avg_coint_pvalue'] = oos_merged['coint_pvalue'].mean()
        summary['oos_avg_hurst'] = oos_merged['hurst'].mean() if 'hurst' in oos_merged else np.nan
        summary['oos_pct_mean_reverting'] = (oos_merged['hurst'] < 0.5).mean() * 100 if 'hurst' in oos_merged else np.nan
        summary['oos_avg_half_life'] = oos_merged['half_life'].mean() if 'half_life' in oos_merged else np.nan
        summary['oos_avg_r_squared'] = oos_merged['r_squared'].mean() if 'r_squared' in oos_merged else np.nan
        summary['oos_hedge_ratio_std'] = oos_merged['hedge_ratio'].std() if 'hedge_ratio' in oos_merged else np.nan
        summary['oos_pct_stationary'] = oos_merged['is_stationary'].mean() * 100 if 'is_stationary' in oos_merged else np.nan

    # --- 9. PnL × Stability Correlations ---
    pnl_corr = {}
    valid = oos_merged.dropna(subset=['rolling_pnl'])
    if len(valid) > 10:
        for col in ['coint_pvalue', 'hurst', 'half_life', 'r_squared', 'adf_pvalue']:
            if col in valid.columns:
                corr = valid['rolling_pnl'].corr(valid[col])
                pnl_corr[col] = round(corr, 3) if not np.isnan(corr) else 0.0

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  OOS SUMMARY")
        print(f"{'=' * 70}")

        if summary:
            pct_coint = summary.get('oos_pct_cointegrated', 0)
            icon = "✅" if pct_coint > 60 else ("⚠️" if pct_coint > 30 else "❌")
            print(f"  {icon} Cointegrated (p<0.05): {pct_coint:.1f}% of OOS period")

            avg_h = summary.get('oos_avg_hurst', 0.5)
            icon = "✅" if avg_h < 0.5 else "❌"
            print(f"  {icon} Avg Hurst: {avg_h:.3f} ({'mean-reverting' if avg_h < 0.5 else 'trending'})")

            mr_pct = summary.get('oos_pct_mean_reverting', 0)
            print(f"    Mean-reverting (H<0.5): {mr_pct:.1f}% of OOS")

            avg_hl = summary.get('oos_avg_half_life', 0)
            icon = "✅" if avg_hl < 20 else ("⚠️" if avg_hl < 40 else "❌")
            print(f"  {icon} Avg Half-Life: {avg_hl:.1f} days")

            hr_std = summary.get('oos_hedge_ratio_std', 0)
            icon = "✅" if hr_std < 0.1 else ("⚠️" if hr_std < 0.3 else "❌")
            print(f"  {icon} Hedge Ratio Stability (σ): {hr_std:.4f}")

            pct_stat = summary.get('oos_pct_stationary', 0)
            icon = "✅" if pct_stat > 60 else "⚠️"
            print(f"  {icon} Spread Stationary (ADF p<0.05): {pct_stat:.1f}% of OOS")

        if pnl_corr:
            print(f"\n  PnL × Stability Correlations:")
            for col, corr in pnl_corr.items():
                direction = "expected" if (
                    (col in ['coint_pvalue', 'hurst', 'half_life', 'adf_pvalue'] and corr < 0)
                    or (col == 'r_squared' and corr > 0)
                ) else "unexpected"
                print(f"    rolling_pnl × {col:.<25} r = {corr:+.3f}  ({direction})")

        # Overall assessment
        print(f"\n{'=' * 70}")
        if summary:
            score = 0
            if summary.get('oos_pct_cointegrated', 0) > 50:
                score += 1
            if summary.get('oos_avg_hurst', 1) < 0.5:
                score += 1
            if summary.get('oos_avg_half_life', 100) < 25:
                score += 1
            if summary.get('oos_hedge_ratio_std', 1) < 0.15:
                score += 1
            if summary.get('oos_pct_stationary', 0) > 50:
                score += 1

            if score >= 4:
                print(f"  ✅ STABLE PAIR ({score}/5 criteria met)")
            elif score >= 2:
                print(f"  ⚠️ PARTIALLY STABLE ({score}/5 criteria met)")
            else:
                print(f"  ❌ UNSTABLE PAIR ({score}/5 criteria met)")

    return {
        'rolling_metrics': merged,
        'summary': summary,
        'pnl_stability_corr': pnl_corr,
        'backtest_data': data,
    }


def run_cointegration_multi_pair(
    pairs: list,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    window: int = 126,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run cointegration analysis across multiple pairs.
    Returns summary DataFrame ranked by pair stability.
    """
    rows = []

    for i, (ty, tx) in enumerate(pairs):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(pairs)}] {ty}/{tx}")
        try:
            result = run_cointegration_analysis(
                ty, tx, start_date, end_date, train_end_date,
                window=window, verbose=False,
            )
            s = result['summary']
            if s:
                s['Pair'] = f"{ty}_{tx}"
                rows.append(s)

                pct = s.get('oos_pct_cointegrated', 0)
                hurst = s.get('oos_avg_hurst', 0.5)
                hl = s.get('oos_avg_half_life', 0)
                icon = "✅" if pct > 50 and hurst < 0.5 else "⚠️"
                print(f"    {icon} Coint={pct:.0f}%  Hurst={hurst:.3f}  HL={hl:.1f}d")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Stability score: higher = more stable
        df['stability_score'] = (
            (df['oos_pct_cointegrated'] / 100) * 0.3 +
            ((1 - df['oos_avg_hurst'].clip(0, 1)) * 0.3) +
            (df['oos_pct_mean_reverting'] / 100 * 0.2) +
            (df['oos_pct_stationary'] / 100 * 0.2)
        )
        df = df.sort_values('stability_score', ascending=False)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  PAIR STABILITY RANKING ({len(df)} pairs)")
            print(f"{'=' * 70}")
            for _, row in df.iterrows():
                bar = "█" * max(1, int(row['stability_score'] * 30))
                print(f"    {row['Pair']:.<20} score={row['stability_score']:.3f}  "
                      f"coint={row['oos_pct_cointegrated']:.0f}%  "
                      f"hurst={row['oos_avg_hurst']:.3f}  {bar}")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    result = run_cointegration_analysis('BAC', 'PNC')

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    result['rolling_metrics'].to_csv(
        os.path.join(results_dir, 'cointegration_rolling_BAC_PNC.csv')
    )

    summary_df = pd.DataFrame([result['summary']])
    summary_df.to_csv(
        os.path.join(results_dir, 'cointegration_summary_BAC_PNC.csv'), index=False
    )

    print(f"\n  Results saved to Research/results/")
