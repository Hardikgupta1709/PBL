"""
Expanded Universe Backtest with Dynamic Pair Selection
========================================================
Week 9: Scale from 3 to 20+ pairs across 10 sectors.

Protocol:
    1. Discover candidate pairs from STOCK_UNIVERSE (min_score ≥ 40)
    2. Run full ConservativeSystem backtest on each pair
    3. Apply the Week 8 dynamic pair health filter
    4. Compare filtered vs unfiltered across the entire universe
    5. Apply Benjamini-Hochberg correction for multiple testing
    6. Generate honest aggregate statistics (no cherry-picking)

Anti-overfit safeguards:
    - Strict temporal split: train ≤ 2020-12-31, OOS > 2020-12-31
    - BH correction on MC p-values (controls false discovery rate)
    - Report MEDIAN Sharpe (robust to outliers), not MEAN
    - Report overfit ratio: train Sharpe / OOS Sharpe
    - No hyperparameter tuning on OOS data
    - Same strategy parameters across ALL pairs (no pair-specific optimization)

Output:
    Research/results/expanded_universe_summary.csv
    Research/results/expanded_universe_report.txt
    Research/results/expanded_filtered_summary.csv
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import yaml
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pair_Discovery.auto_find_pairs import (
    STOCK_UNIVERSE, download_prices, quick_pair_score
)
from Core_Strategy.conservative_strategy import (
    ConservativeSystem, download_data, calculate_half_life
)
from Core_Strategy.strategy_validator import (
    MonteCarloValidator, BootstrapAnalyzer, benjamini_hochberg
)
from Research.dynamic_pair_selector import (
    PairHealthMonitor, HealthThresholds, backtest_with_dynamic_filter,
    apply_health_gate, DynamicPairSelector
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

def load_config() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config_experiments.yaml'
    )
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def check_universe_data_readiness(
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    min_data_days: int = 504,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Baseline data readiness check for the expanded stock universe.

    Reports per-ticker coverage and flags insufficient history.
    """
    rows = []
    for sector, stocks in STOCK_UNIVERSE.items():
        for ticker in stocks:
            data = download_prices(ticker, start_date, end_date)
            n_days = len(data) if data is not None else 0
            start = data.index.min() if data is not None and n_days > 0 else None
            end = data.index.max() if data is not None and n_days > 0 else None
            rows.append({
                'Sector': sector,
                'Ticker': ticker,
                'Data_Days': n_days,
                'Start_Date': str(start.date()) if start is not None else '',
                'End_Date': str(end.date()) if end is not None else '',
                'Sufficient_Data': n_days >= min_data_days,
            })

    df = pd.DataFrame(rows)
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'universe_data_readiness.csv')
    df.to_csv(out_path, index=False)

    if verbose:
        total = len(df)
        ok = int(df['Sufficient_Data'].sum())
        print(f"\n{'=' * 70}")
        print("  DATA READINESS CHECK")
        print(f"{'=' * 70}")
        print(f"  Tickers checked: {total}")
        print(f"  Sufficient data: {ok}/{total}")
        print(f"  Saved: {out_path}")

    return df


# =============================================================================
# EXPANDED PAIR DISCOVERY
# =============================================================================

def discover_expanded_universe(
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    min_score: int = 40,
    max_pairs_per_sector: int = 5,
    min_sector_pairs: int = 1,
    min_sector_median_score: float = 0.0,
    min_data_days: int = 504,
    min_adv: float = 10_000_000.0,
    max_spread_vol: float = 0.08,
    coint_p_threshold: float = 0.20,
    require_coint_stability: bool = True,
    target_pairs: int = 25,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Discover 20-60 pairs across all 10 sectors.

    Strategy:
        - Scan the expanded stock universe across 10 sectors
        - Score each intra-sector pair with quick_pair_score()
        - Select top pairs per sector (balanced representation)
        - Sort by score, take top `target_pairs`

    This ensures sector diversity — not just banking pairs.
    """
    all_pairs = []

    for sector, stocks in STOCK_UNIVERSE.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Scanning: {sector} ({len(stocks)} stocks)")
            print(f"{'=' * 60}")

        # Download all prices for sector
        stock_data = {}
        stock_adv = {}
        for ticker in stocks:
            try:
                raw = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
            except Exception:
                raw = None

            if raw is None or raw.empty:
                if verbose:
                    print(f"    ✗ {ticker} (insufficient data)")
                continue

            price = raw['Close'] if 'Close' in raw else raw.iloc[:, 0]
            if isinstance(price, pd.DataFrame):
                price = price.iloc[:, 0]
            price = price.dropna()
            if not isinstance(price, pd.Series):
                if verbose:
                    print(f"    ✗ {ticker} (invalid price series)")
                continue
            volume = raw['Volume'] if 'Volume' in raw else pd.Series(dtype=float)
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            volume = volume.dropna()
            if len(price) < min_data_days:
                if verbose:
                    print(f"    ✗ {ticker} (insufficient data)")
                continue

            stock_data[ticker] = price
            if len(volume) > 0:
                adv_val = (price * volume).mean()
                adv = float(adv_val) if np.isscalar(adv_val) else float(adv_val.iloc[0])
            else:
                adv = 0.0
            stock_adv[ticker] = adv

            if verbose:
                print(f"    ✓ {ticker} ({len(price)} days)")

        if len(stock_data) < 2:
            continue

        # Score all intra-sector pairs
        pair_scores = []
        sector_pairs = list(combinations(stock_data.keys(), 2))

        for t1, t2 in sector_pairs:
            score, metrics = quick_pair_score(stock_data[t1], stock_data[t2])
            if score >= min_score:
                adv_pair = min(stock_adv.get(t1, 0.0), stock_adv.get(t2, 0.0))
                if adv_pair < min_adv:
                    continue

                # Cointegration stability (train split halves)
                if require_coint_stability:
                    prices_y = stock_data[t1]
                    prices_x = stock_data[t2]
                    train_end = pd.Timestamp('2020-12-31')
                    train = pd.DataFrame({'Y': prices_y, 'X': prices_x}).dropna()
                    train = train[train.index <= train_end]
                    if len(train) < min_data_days:
                        continue
                    mid = len(train) // 2
                    try:
                        from statsmodels.tsa.stattools import coint
                        _, p1, _ = coint(train['Y'].iloc[:mid], train['X'].iloc[:mid])
                        _, p2, _ = coint(train['Y'].iloc[mid:], train['X'].iloc[mid:])
                        if p1 > coint_p_threshold or p2 > coint_p_threshold:
                            continue
                    except Exception:
                        continue

                # Spread volatility screen (train period)
                try:
                    hedge = metrics.get('hedge_ratio', 1.0)
                    ret_y = stock_data[t1].pct_change().dropna()
                    ret_x = stock_data[t2].pct_change().dropna()
                    common = ret_y.index.intersection(ret_x.index)
                    ret_y = ret_y.loc[common]
                    ret_x = ret_x.loc[common]
                    train_mask = ret_y.index <= pd.Timestamp('2020-12-31')
                    spread_ret = ret_y[train_mask] - hedge * ret_x[train_mask]
                    if spread_ret.std() > max_spread_vol:
                        continue
                except Exception:
                    continue

                pair_scores.append({
                    'Sector': sector,
                    'Ticker_Y': t1,
                    'Ticker_X': t2,
                    'Score': score,
                    'Coint_pval': metrics.get('coint_pval', 1),
                    'Correlation': metrics.get('correlation', 0),
                    'ADF_pval': metrics.get('adf_pval', 1),
                    'Half_Life': metrics.get('half_life', 999),
                    'Hedge_Ratio': metrics.get('hedge_ratio', 0),
                    'ADV_Min': adv_pair,
                })

        # Keep top pairs per sector with optional sector constraints
        sector_df = pd.DataFrame(pair_scores)
        if len(sector_df) > 0:
            sector_df = sector_df.sort_values('Score', ascending=False)
            sector_df = sector_df.head(max_pairs_per_sector)

            if len(sector_df) < min_sector_pairs:
                if verbose:
                    print(f"  ⚠ Skipping {sector}: only {len(sector_df)} pairs")
                continue

            median_score = float(sector_df['Score'].median())
            if median_score < min_sector_median_score:
                if verbose:
                    print(f"  ⚠ Skipping {sector}: median score {median_score:.1f} < {min_sector_median_score:.1f}")
                continue

            all_pairs.append(sector_df)
            if verbose:
                print(f"  → Found {len(sector_df)} viable pairs in {sector}")

        # Rate limit between sectors
        time.sleep(0.5)

    if len(all_pairs) == 0:
        return pd.DataFrame()

    result = pd.concat(all_pairs, ignore_index=True)
    result = result.sort_values('Score', ascending=False).reset_index(drop=True)

    # If we have more than target, trim (keeping sector balance)
    if len(result) > target_pairs:
        # Ensure at least 1 pair per represented sector
        balanced = []
        sectors = result['Sector'].unique()
        per_sector = max(1, target_pairs // len(sectors))
        remainder = target_pairs - per_sector * len(sectors)

        for sector in sectors:
            sector_rows = result[result['Sector'] == sector]
            take = min(per_sector, len(sector_rows))
            balanced.append(sector_rows.head(take))

        result = pd.concat(balanced, ignore_index=True)
        result = result.sort_values('Score', ascending=False).reset_index(drop=True)

        # Fill remaining slots with highest scores not yet included
        if len(result) < target_pairs:
            all_sorted = pd.concat(all_pairs, ignore_index=True).sort_values('Score', ascending=False)
            for _, row in all_sorted.iterrows():
                key = f"{row['Ticker_Y']}_{row['Ticker_X']}"
                existing = result.apply(
                    lambda r: f"{r['Ticker_Y']}_{r['Ticker_X']}", axis=1
                )
                if key not in existing.values and len(result) < target_pairs:
                    result = pd.concat(
                        [result, pd.DataFrame([row])], ignore_index=True
                    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  TOTAL PAIRS SELECTED: {len(result)}")
        print(f"  Sectors: {result['Sector'].nunique()}")
        print(f"  Score range: [{result['Score'].min()}, {result['Score'].max()}]")
        print(f"{'=' * 60}")
        for sector in result['Sector'].unique():
            n = len(result[result['Sector'] == sector])
            print(f"    {sector}: {n} pair{'s' if n > 1 else ''}")

    return result


# =============================================================================
# UNIVERSE BACKTEST WITH DYNAMIC FILTER
# =============================================================================

def backtest_expanded_universe(
    pairs_df: pd.DataFrame,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    run_dynamic_filter: bool = True,
    apply_health_sizing: bool = False,
    sizing_thresholds: Tuple[float, float, float] = (0.5, 0.7, 0.85),
    run_monte_carlo: bool = True,
    run_bootstrap: bool = True,
    health_thresholds: Optional[HealthThresholds] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run backtest on every discovered pair with optional dynamic filter.

    For each pair, computes:
        - Standard (unfiltered) OOS metrics
        - Filtered OOS metrics (if run_dynamic_filter=True)
        - Monte Carlo p-value
        - Bootstrap Sharpe CI
    """
    results = []
    n_pairs = len(pairs_df)

    for idx, row in pairs_df.iterrows():
        ty, tx = row['Ticker_Y'], row['Ticker_X']
        pair_name = f"{ty}_{tx}"

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  [{idx+1}/{n_pairs}] Backtesting {pair_name} "
                  f"(Score={row['Score']}, Sector={row['Sector']})")

        try:
            # Download data
            stock_y = download_prices(ty, start_date, end_date)
            stock_x = download_prices(tx, start_date, end_date)
            market = download_prices('SPY', start_date, end_date)

            if stock_y is None or stock_x is None or market is None:
                if verbose:
                    print(f"    ✗ Data download failed")
                continue

            # --- Run UNFILTERED backtest ---
            system = ConservativeSystem()
            bt = system.run_backtest(
                stock_y, stock_x, market,
                train_end_date=train_end_date,
                slippage_bps=slippage_bps,
                verbose=False,
            )

            oos = bt[bt['period'] == 'oos']
            train = bt[bt['period'] == 'train']
            oos_ret = oos['strategy_return'].dropna()
            train_ret = train['strategy_return'].dropna()

            if len(oos_ret) < 50:
                if verbose:
                    print(f"    ✗ Insufficient OOS data ({len(oos_ret)} days)")
                continue

            # Unfiltered OOS metrics
            oos_total = (1 + oos_ret).prod() - 1
            oos_vol = oos_ret.std() * np.sqrt(252)
            oos_ann = (1 + oos_total) ** (252 / len(oos_ret)) - 1
            oos_sharpe = oos_ann / oos_vol if oos_vol > 0 else 0

            train_total = (1 + train_ret).prod() - 1
            train_vol = train_ret.std() * np.sqrt(252)
            train_ann = (1 + train_total) ** (252 / max(len(train_ret), 1)) - 1
            train_sharpe = train_ann / train_vol if train_vol > 0 else 0

            # Market benchmark
            mkt_ret = oos['market_return'].dropna()
            mkt_total = (1 + mkt_ret).prod() - 1

            # Trade analysis
            trades = system.get_trade_analysis(period='oos')
            n_trades = len(trades)
            win_rate = trades['profitable'].mean() if n_trades > 0 else 0
            avg_pnl = trades['pnl_pct'].mean() if n_trades > 0 else 0

            # Max drawdown
            cum = (1 + oos_ret).cumprod()
            max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()

            record = {
                'Pair': pair_name,
                'Sector': row['Sector'],
                'Discovery_Score': row['Score'],
                'Coint_pval': row['Coint_pval'],
                'Half_Life': row['Half_Life'],
                'Train_Return_%': train_total * 100,
                'Train_Sharpe': train_sharpe,
                'UF_OOS_Return_%': oos_total * 100,
                'UF_OOS_Sharpe': oos_sharpe,
                'UF_OOS_MaxDD_%': max_dd * 100,
                'UF_OOS_Trades': n_trades,
                'UF_OOS_WinRate_%': win_rate * 100,
                'UF_OOS_AvgPnL_%': avg_pnl,
                'Market_Return_%': mkt_total * 100,
                'UF_Alpha_%': (oos_total - mkt_total) * 100,
            }

            # --- Run FILTERED backtest ---
            if run_dynamic_filter:
                try:
                    filt_result = backtest_with_dynamic_filter(
                        stock_y, stock_x, market,
                        train_end_date=train_end_date,
                        thresholds=health_thresholds,
                        health_step=5,
                        apply_health_sizing=apply_health_sizing,
                        sizing_thresholds=sizing_thresholds,
                        verbose=False,
                        slippage_bps=slippage_bps,
                    )

                    mf = filt_result['metrics_filtered']['oos']
                    fs = filt_result['filter_stats']

                    record['F_OOS_Return_%'] = mf['total_return'] * 100
                    record['F_OOS_Sharpe'] = mf['sharpe']
                    record['F_OOS_MaxDD_%'] = mf['max_dd'] * 100
                    record['F_OOS_Trades'] = mf['n_trades']
                    record['F_OOS_WinRate_%'] = mf['win_rate'] * 100
                    record['F_Alpha_%'] = (mf['total_return'] - mkt_total) * 100
                    record['Filter_Sharpe_Change'] = fs['sharpe_change']
                    record['Pct_Healthy'] = fs['oos_pct_healthy']
                    record['Avg_Health_Score'] = fs['oos_avg_health_score']
                    record['Signals_Blocked_%'] = fs['signals_blocked_pct']
                    record['Filter_Improved'] = fs['sharpe_change'] > 0
                    if 'oos_avg_position_size' in fs:
                        record['F_OOS_AvgPosSize'] = fs['oos_avg_position_size']

                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Filter error: {e}")
                    record['F_OOS_Sharpe'] = np.nan
                    record['Filter_Improved'] = False

            # --- Monte Carlo p-value ---
            if run_monte_carlo and n_trades >= 3:
                try:
                    mc = MonteCarloValidator(
                        n_simulations=1000, random_seed=42 + idx
                    )
                    mc_result = mc.run(system, period='oos', verbose=False)
                    record['MC_pvalue'] = mc_result['p_value_sharpe']
                except Exception:
                    record['MC_pvalue'] = np.nan
            else:
                record['MC_pvalue'] = np.nan

            # --- Bootstrap CI ---
            if run_bootstrap and len(oos_ret) >= 30:
                try:
                    bs = BootstrapAnalyzer(n_resamples=1000, random_seed=42 + idx)
                    bs_result = bs.run(oos_ret, verbose=False)
                    record['BS_Sharpe_Lo'] = bs_result['sharpe_ci_lower']
                    record['BS_Sharpe_Hi'] = bs_result['sharpe_ci_upper']
                except Exception:
                    record['BS_Sharpe_Lo'] = np.nan
                    record['BS_Sharpe_Hi'] = np.nan
            else:
                record['BS_Sharpe_Lo'] = np.nan
                record['BS_Sharpe_Hi'] = np.nan

            results.append(record)

            if verbose:
                mc_str = ""
                if not np.isnan(record.get('MC_pvalue', np.nan)):
                    mc_str = f" MC-p={record['MC_pvalue']:.3f}"

                filt_str = ""
                if run_dynamic_filter and not np.isnan(record.get('F_OOS_Sharpe', np.nan)):
                    delta = record.get('Filter_Sharpe_Change', 0)
                    icon = "↑" if delta > 0 else "↓"
                    filt_str = f" | Filt: {record['F_OOS_Sharpe']:.2f} ({icon}{abs(delta):.2f})"

                print(f"    ✓ UF: Ret={oos_total*100:+.2f}% "
                      f"Sharpe={oos_sharpe:.2f} "
                      f"Trades={n_trades}{mc_str}{filt_str}")

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            continue

        # Rate limit
        time.sleep(0.3)

    return pd.DataFrame(results)


# =============================================================================
# AGGREGATE ANALYSIS (EXPANDED)
# =============================================================================

def expanded_aggregate_analysis(
    results_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Compute aggregate statistics with anti-overfit safeguards.

    Reports both unfiltered and filtered metrics side by side.
    Uses median (not mean) for robustness.
    """
    if len(results_df) == 0:
        return {}

    n = len(results_df)

    # --- Unfiltered stats ---
    uf_profitable = results_df[results_df['UF_OOS_Return_%'] > 0]
    uf_positive_alpha = results_df[results_df['UF_Alpha_%'] > 0]

    agg = {
        'Total Pairs Tested': n,
        'Sectors Represented': results_df['Sector'].nunique(),
    }

    # Unfiltered
    agg['UF_Profitable_Pairs'] = len(uf_profitable)
    agg['UF_Hit_Rate_%'] = len(uf_profitable) / n * 100
    agg['UF_Positive_Alpha'] = len(uf_positive_alpha)
    agg['UF_Median_Sharpe'] = results_df['UF_OOS_Sharpe'].median()
    agg['UF_Mean_Sharpe'] = results_df['UF_OOS_Sharpe'].mean()
    agg['UF_Median_Return_%'] = results_df['UF_OOS_Return_%'].median()
    agg['UF_Median_MaxDD_%'] = results_df['UF_OOS_MaxDD_%'].median()
    agg['UF_Avg_Trades'] = results_df['UF_OOS_Trades'].mean()

    # Filtered (if available)
    has_filter = 'F_OOS_Sharpe' in results_df.columns
    if has_filter:
        filt = results_df.dropna(subset=['F_OOS_Sharpe'])
        if len(filt) > 0:
            f_profitable = filt[filt['F_OOS_Return_%'] > 0]
            agg['F_Profitable_Pairs'] = len(f_profitable)
            agg['F_Hit_Rate_%'] = len(f_profitable) / len(filt) * 100
            agg['F_Median_Sharpe'] = filt['F_OOS_Sharpe'].median()
            agg['F_Mean_Sharpe'] = filt['F_OOS_Sharpe'].mean()
            agg['F_Median_Return_%'] = filt['F_OOS_Return_%'].median()
            agg['F_Median_MaxDD_%'] = filt['F_OOS_MaxDD_%'].median()
            agg['F_Avg_Trades'] = filt['F_OOS_Trades'].mean()

            # Filter effectiveness
            improved = filt[filt.get('Filter_Improved', False) == True]
            agg['Filter_Improved_Count'] = len(improved)
            agg['Filter_Improved_%'] = len(improved) / len(filt) * 100
            agg['Avg_Sharpe_Change'] = filt['Filter_Sharpe_Change'].mean()
            agg['Avg_Pct_Healthy'] = filt['Pct_Healthy'].mean()

    # Overfit analysis
    agg['Train_Mean_Sharpe'] = results_df['Train_Sharpe'].mean()
    agg['OOS_Mean_Sharpe'] = results_df['UF_OOS_Sharpe'].mean()
    agg['Overfit_Ratio'] = (
        agg['Train_Mean_Sharpe'] / agg['OOS_Mean_Sharpe']
        if agg['OOS_Mean_Sharpe'] != 0 else float('inf')
    )
    agg['Train→OOS_Sharpe_Drop'] = (
        f"{agg['Train_Mean_Sharpe']:.3f} → {agg['OOS_Mean_Sharpe']:.3f}"
    )

    # Sector breakdown
    sector_stats = results_df.groupby('Sector').agg({
        'UF_OOS_Sharpe': ['median', 'mean', 'count'],
        'UF_OOS_Return_%': 'median',
    }).round(3)
    agg['Sector_Breakdown'] = sector_stats

    # BH correction on MC p-values
    mc_pvals = results_df[['Pair', 'MC_pvalue']].dropna()
    if len(mc_pvals) > 0:
        bh_results = benjamini_hochberg(
            list(zip(mc_pvals['Pair'], mc_pvals['MC_pvalue'])),
            alpha=0.05
        )
        n_sig_raw = sum(1 for _, p, _, _ in bh_results if p < 0.05)
        n_sig_bh = sum(1 for _, _, _, sig in bh_results if sig)
        agg['MC_Significant_Raw'] = n_sig_raw
        agg['MC_Significant_BH'] = n_sig_bh
        agg['BH_Details'] = bh_results
    else:
        agg['MC_Significant_Raw'] = 0
        agg['MC_Significant_BH'] = 0

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  EXPANDED UNIVERSE — AGGREGATE STATISTICS")
        print(f"{'=' * 70}")

        print(f"\n  Universe: {n} pairs across {agg['Sectors Represented']} sectors")
        print(f"  Overfit check: {agg['Train→OOS_Sharpe_Drop']}")

        print(f"\n  {'Metric':<35} {'Unfiltered':>12}", end='')
        if has_filter:
            print(f" {'Filtered':>12}", end='')
        print()
        print(f"  {'─' * 59}")

        metric_pairs = [
            ('Profitable Pairs', 'UF_Profitable_Pairs', 'F_Profitable_Pairs', 'd'),
            ('Hit Rate (%)', 'UF_Hit_Rate_%', 'F_Hit_Rate_%', '.1f'),
            ('Median OOS Sharpe', 'UF_Median_Sharpe', 'F_Median_Sharpe', '.3f'),
            ('Mean OOS Sharpe', 'UF_Mean_Sharpe', 'F_Mean_Sharpe', '.3f'),
            ('Median OOS Return (%)', 'UF_Median_Return_%', 'F_Median_Return_%', '.2f'),
            ('Median Max DD (%)', 'UF_Median_MaxDD_%', 'F_Median_MaxDD_%', '.2f'),
            ('Avg OOS Trades', 'UF_Avg_Trades', 'F_Avg_Trades', '.1f'),
        ]

        for label, uf_key, f_key, fmt in metric_pairs:
            uf_val = agg.get(uf_key, 0)
            print(f"  {label:<35} {uf_val:>12{fmt}}", end='')
            if has_filter and f_key in agg:
                f_val = agg.get(f_key, 0)
                print(f" {f_val:>12{fmt}}", end='')
            print()

        if has_filter:
            print(f"\n  Filter Stats:")
            print(f"    Pairs improved: {agg.get('Filter_Improved_Count', 0)}/{n}")
            print(f"    Avg Sharpe change: {agg.get('Avg_Sharpe_Change', 0):+.3f}")
            print(f"    Avg pair health: {agg.get('Avg_Pct_Healthy', 0):.1f}%")

        print(f"\n  BH Multiple Testing Correction:")
        print(f"    Significant (raw p<0.05): {agg['MC_Significant_Raw']}/{n}")
        print(f"    Significant (BH adj):     {agg['MC_Significant_BH']}/{n}")

        print(f"\n  SECTOR BREAKDOWN:")
        print(sector_stats.to_string())

    return agg


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_expanded_report(
    results_df: pd.DataFrame,
    agg: dict,
    save_dir: Optional[str] = None,
) -> str:
    """Generate full expanded universe report."""
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(save_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(save_dir, 'expanded_universe_summary.csv')
    results_df.to_csv(csv_path, index=False)

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("EXPANDED UNIVERSE BACKTEST REPORT (Week 9)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Pairs: {len(results_df)} | Sectors: {results_df['Sector'].nunique()}")
    lines.append("=" * 80)
    lines.append("")

    # Key metrics
    lines.append("KEY METRICS (Anti-Overfit)")
    lines.append("-" * 80)
    for k, v in agg.items():
        if k in ('Sector_Breakdown', 'BH_Details'):
            continue
        lines.append(f"  {k:.<55} {v}")
    lines.append("")

    # Top pairs by OOS Sharpe (unfiltered)
    lines.append("TOP 10 PAIRS — UNFILTERED OOS SHARPE")
    lines.append("-" * 80)
    top = results_df.nlargest(10, 'UF_OOS_Sharpe')
    for _, row in top.iterrows():
        lines.append(
            f"  {row['Pair']:.<18} Sharpe={row['UF_OOS_Sharpe']:.3f}  "
            f"Ret={row['UF_OOS_Return_%']:+.2f}%  "
            f"Alpha={row['UF_Alpha_%']:+.2f}%  "
            f"Trades={int(row['UF_OOS_Trades'])}  "
            f"Sector={row['Sector']}"
        )
    lines.append("")

    # Top pairs by FILTERED Sharpe
    if 'F_OOS_Sharpe' in results_df.columns:
        filt = results_df.dropna(subset=['F_OOS_Sharpe'])
        if len(filt) > 0:
            lines.append("TOP 10 PAIRS — FILTERED OOS SHARPE")
            lines.append("-" * 80)
            top_f = filt.nlargest(10, 'F_OOS_Sharpe')
            for _, row in top_f.iterrows():
                lines.append(
                    f"  {row['Pair']:.<18} Sharpe={row['F_OOS_Sharpe']:.3f}  "
                    f"Ret={row['F_OOS_Return_%']:+.2f}%  "
                    f"Healthy={row['Pct_Healthy']:.0f}%  "
                    f"Change={row['Filter_Sharpe_Change']:+.3f}"
                )
            lines.append("")

    # Bottom pairs
    lines.append("BOTTOM 5 PAIRS — UNFILTERED OOS SHARPE")
    lines.append("-" * 80)
    bot = results_df.nsmallest(5, 'UF_OOS_Sharpe')
    for _, row in bot.iterrows():
        lines.append(
            f"  {row['Pair']:.<18} Sharpe={row['UF_OOS_Sharpe']:.3f}  "
            f"Ret={row['UF_OOS_Return_%']:+.2f}%  "
            f"Sector={row['Sector']}"
        )
    lines.append("")

    # Honesty section
    lines.append("=" * 80)
    lines.append("HONESTY ASSESSMENT")
    lines.append("=" * 80)
    hit = agg.get('UF_Hit_Rate_%', 0)
    med_sh = agg.get('UF_Median_Sharpe', 0)
    overfit = agg.get('Overfit_Ratio', float('inf'))
    bh_sig = agg.get('MC_Significant_BH', 0)
    total = agg.get('Total Pairs Tested', 1)

    if med_sh > 0.3 and hit > 60:
        lines.append("✅ STRONG: Positive median Sharpe across 60%+ of universe")
    elif med_sh > 0 and hit > 40:
        lines.append("⚠️ MODERATE: Some pairs profitable, but not majority")
    else:
        lines.append("❌ WEAK: Strategy does not generalize well across pairs")
        lines.append("   (This is an honest assessment — negative results are valid)")

    lines.append(f"   Overfit ratio (train/OOS): {overfit:.1f}x")
    lines.append(f"   BH-corrected significant: {bh_sig}/{total}")

    if overfit > 5:
        lines.append("   ⚠️ HIGH OVERFIT: Train Sharpe >> OOS Sharpe")
    elif overfit > 2:
        lines.append("   ⚠️ MODERATE OVERFIT: Some degradation expected")
    else:
        lines.append("   ✅ LOW OVERFIT: Train/OOS Sharpes are comparable")

    report_text = "\n".join(lines)

    report_path = os.path.join(save_dir, 'expanded_universe_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n  Reports saved:")
    print(f"    CSV:  {csv_path}")
    print(f"    Text: {report_path}")

    return report_text


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_expanded_universe_backtest(
    min_score: int = 40,
    max_pairs_per_sector: int = 5,
    min_sector_pairs: int = 1,
    min_sector_median_score: float = 0.0,
    target_pairs: int = 25,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    run_dynamic_filter: bool = True,
    apply_health_sizing: bool = False,
    sizing_thresholds: Tuple[float, float, float] = (0.5, 0.7, 0.85),
    run_mc: bool = True,
    run_bs: bool = True,
    run_data_check: bool = False,
    min_data_days: int = 504,
    min_adv: float = 10_000_000.0,
    max_spread_vol: float = 0.08,
    require_coint_stability: bool = True,
    coint_p_threshold: float = 0.20,
) -> Tuple[pd.DataFrame, dict]:
    """
    Full pipeline: discover 20+ pairs → backtest → filter → aggregate → report.
    """
    print("=" * 70)
    print("  EXPANDED UNIVERSE BACKTEST — WEEK 9")
    print(f"  {start_date} → {end_date} | Train ≤ {train_end_date}")
    print(f"  Target: {target_pairs} pairs | Min score: {min_score}")
    print(f"  Sector guardrails: min pairs {min_sector_pairs}, median score ≥ {min_sector_median_score}")
    print(f"  Dynamic filter: {'ON' if run_dynamic_filter else 'OFF'}")
    if run_dynamic_filter and apply_health_sizing:
        low, mid, high = sizing_thresholds
        print(f"  Health-aware sizing: ON ({low:.2f}, {mid:.2f}, {high:.2f})")
    print("=" * 70)

    if run_data_check:
        check_universe_data_readiness(
            start_date=start_date,
            end_date=end_date,
            min_data_days=min_data_days,
            verbose=True,
        )

    # Step 1: Discover pairs
    print(f"\n📡 STEP 1: EXPANDED PAIR DISCOVERY")
    pairs = discover_expanded_universe(
        start_date=start_date,
        end_date=end_date,
        min_score=min_score,
        max_pairs_per_sector=max_pairs_per_sector,
        min_sector_pairs=min_sector_pairs,
        min_sector_median_score=min_sector_median_score,
        target_pairs=target_pairs,
        min_data_days=min_data_days,
        min_adv=min_adv,
        max_spread_vol=max_spread_vol,
        require_coint_stability=require_coint_stability,
        coint_p_threshold=coint_p_threshold,
    )

    if len(pairs) == 0:
        print("  ✗ No viable pairs found!")
        return pd.DataFrame(), {}

    # Save discovered pairs
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    pairs.to_csv(
        os.path.join(results_dir, 'expanded_discovered_pairs.csv'), index=False
    )

    # Step 2: Backtest all pairs
    print(f"\n📊 STEP 2: BACKTESTING {len(pairs)} PAIRS")
    bt_results = backtest_expanded_universe(
        pairs,
        start_date=start_date,
        end_date=end_date,
        train_end_date=train_end_date,
        slippage_bps=slippage_bps,
        run_dynamic_filter=run_dynamic_filter,
        apply_health_sizing=apply_health_sizing,
        sizing_thresholds=sizing_thresholds,
        run_monte_carlo=run_mc,
        run_bootstrap=run_bs,
    )

    if len(bt_results) == 0:
        print("  ✗ All backtests failed!")
        return pd.DataFrame(), {}

    # Step 3: Aggregate analysis
    print(f"\n📈 STEP 3: AGGREGATE ANALYSIS")
    agg = expanded_aggregate_analysis(bt_results)

    # Step 4: Generate reports
    print(f"\n📝 STEP 4: GENERATING REPORT")
    generate_expanded_report(bt_results, agg)

    return bt_results, agg


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Week 9: Expanded universe backtest with dynamic filter'
    )
    parser.add_argument('--min-score', type=int, default=40,
                        help='Minimum pair discovery score')
    parser.add_argument('--max-pairs-per-sector', type=int, default=5,
                        help='Max pairs per sector')
    parser.add_argument('--min-sector-pairs', type=int, default=1,
                        help='Minimum pairs required to keep a sector')
    parser.add_argument('--min-sector-median-score', type=float, default=0.0,
                        help='Minimum sector median score to keep the sector')
    parser.add_argument('--target', type=int, default=25,
                        help='Target number of pairs')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip dynamic filter')
    parser.add_argument('--health-sizing', action='store_true',
                        help='Enable health-aware position sizing')
    parser.add_argument('--size-low', type=float, default=0.5,
                        help='Health sizing low threshold')
    parser.add_argument('--size-mid', type=float, default=0.7,
                        help='Health sizing mid threshold')
    parser.add_argument('--size-high', type=float, default=0.85,
                        help='Health sizing high threshold')
    parser.add_argument('--no-mc', action='store_true',
                        help='Skip Monte Carlo validation')
    parser.add_argument('--no-bs', action='store_true',
                        help='Skip Bootstrap analysis')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer pairs, no MC/BS')
    parser.add_argument('--data-check', action='store_true',
                        help='Run baseline data readiness check')
    parser.add_argument('--min-adv', type=float, default=10_000_000.0,
                        help='Minimum average daily dollar volume')
    parser.add_argument('--max-spread-vol', type=float, default=0.08,
                        help='Maximum train spread volatility')
    parser.add_argument('--no-coint-stability', action='store_true',
                        help='Disable cointegration stability screen')
    parser.add_argument('--coint-p-thresh', type=float, default=0.20,
                        help='Cointegration p-value threshold for stability')

    args = parser.parse_args()

    if args.quick:
        results, aggregate = run_expanded_universe_backtest(
            min_score=50,
            max_pairs_per_sector=2,
            min_sector_pairs=1,
            min_sector_median_score=0.0,
            target_pairs=10,
            run_dynamic_filter=not args.no_filter,
            apply_health_sizing=args.health_sizing,
            sizing_thresholds=(args.size_low, args.size_mid, args.size_high),
            run_mc=False,
            run_bs=False,
            run_data_check=args.data_check,
            min_adv=args.min_adv,
            max_spread_vol=args.max_spread_vol,
            require_coint_stability=not args.no_coint_stability,
            coint_p_threshold=args.coint_p_thresh,
        )
    else:
        results, aggregate = run_expanded_universe_backtest(
            min_score=args.min_score,
            max_pairs_per_sector=args.max_pairs_per_sector,
            min_sector_pairs=args.min_sector_pairs,
            min_sector_median_score=args.min_sector_median_score,
            target_pairs=args.target,
            run_dynamic_filter=not args.no_filter,
            apply_health_sizing=args.health_sizing,
            sizing_thresholds=(args.size_low, args.size_mid, args.size_high),
            run_mc=not args.no_mc,
            run_bs=not args.no_bs,
            run_data_check=args.data_check,
            min_adv=args.min_adv,
            max_spread_vol=args.max_spread_vol,
            require_coint_stability=not args.no_coint_stability,
            coint_p_threshold=args.coint_p_thresh,
        )

    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(results)} pairs backtested")
    print(f"{'=' * 70}")
