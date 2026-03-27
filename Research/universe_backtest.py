"""
Universe Backtest: Scale Strategy Across 50+ Pairs
====================================================
Scans all 10 sectors from the stock universe, discovers viable pairs,
runs the full backtest + validation on each, and aggregates statistics.

This addresses Improvement 3 from the research upgrade plan:
    - "Is the strategy a one-pair wonder?"
    - Aggregate Sharpe, hit rate, median return across the whole universe
    - Benjamini-Hochberg correction for multiple testing
    - Survivorship-bias-aware pair selection

Output:
    Research/results/universe_summary.csv — per-pair metrics
    Research/results/universe_report.txt  — aggregate report
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import yaml
from datetime import datetime

# Add project root
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config_experiments.yaml'
    )
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# =============================================================================
# PAIR DISCOVERY
# =============================================================================

def discover_pairs(start_date: str, end_date: str,
                   min_score: int = 40,
                   max_pairs_per_sector: int = 10,
                   min_data_days: int = 504,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Scan all sectors and return scored pairs above threshold.
    """
    all_pairs = []

    for sector, stocks in STOCK_UNIVERSE.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Scanning: {sector} ({len(stocks)} stocks)")
            print(f"{'=' * 60}")

        # Download all stock data for this sector
        stock_data = {}
        for ticker in stocks:
            data = download_prices(ticker, start_date, end_date)
            if data is not None and len(data) >= min_data_days:
                stock_data[ticker] = data
                if verbose:
                    print(f"    ✓ {ticker} ({len(data)} days)")
            else:
                if verbose:
                    print(f"    ✗ {ticker} (insufficient data)")

        if len(stock_data) < 2:
            continue

        # Score all pairs
        from itertools import combinations
        pair_scores = []

        for t1, t2 in combinations(stock_data.keys(), 2):
            score, metrics = quick_pair_score(stock_data[t1], stock_data[t2])
            if score >= min_score:
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
                })

        # Keep top pairs per sector
        sector_df = pd.DataFrame(pair_scores)
        if len(sector_df) > 0:
            sector_df = sector_df.sort_values('Score', ascending=False)
            sector_df = sector_df.head(max_pairs_per_sector)
            all_pairs.append(sector_df)
            if verbose:
                print(f"  → Found {len(sector_df)} viable pairs in {sector}")

    if len(all_pairs) == 0:
        return pd.DataFrame()

    result = pd.concat(all_pairs, ignore_index=True)
    result = result.sort_values('Score', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  TOTAL VIABLE PAIRS: {len(result)}")
        print(f"  Score range: [{result['Score'].min()}, {result['Score'].max()}]")
        print(f"{'=' * 60}")

    return result


# =============================================================================
# UNIVERSE BACKTEST
# =============================================================================

def backtest_universe(pairs_df: pd.DataFrame,
                      start_date: str = '2015-01-01',
                      end_date: str = '2025-06-30',
                      train_end_date: str = '2020-12-31',
                      slippage_bps: float = 5.0,
                      run_monte_carlo: bool = True,
                      run_bootstrap: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Run backtest on every discovered pair and collect metrics.
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

            # Run backtest
            system = ConservativeSystem()
            bt = system.run_backtest(
                stock_y, stock_x, market,
                train_end_date=train_end_date,
                slippage_bps=slippage_bps,
                verbose=False,
            )

            # Compute OOS metrics
            oos = bt[bt['period'] == 'oos']
            train = bt[bt['period'] == 'train']
            oos_ret = oos['strategy_return'].dropna()
            train_ret = train['strategy_return'].dropna()

            if len(oos_ret) < 50:
                if verbose:
                    print(f"    ✗ Insufficient OOS data ({len(oos_ret)} days)")
                continue

            oos_total = (1 + oos_ret).prod() - 1
            oos_vol = oos_ret.std() * np.sqrt(252)
            oos_ann = (1 + oos_total) ** (252 / len(oos_ret)) - 1
            oos_sharpe = oos_ann / oos_vol if oos_vol > 0 else 0

            train_total = (1 + train_ret).prod() - 1
            train_vol = train_ret.std() * np.sqrt(252)
            train_ann = (1 + train_total) ** (252 / max(len(train_ret), 1)) - 1
            train_sharpe = train_ann / train_vol if train_vol > 0 else 0

            # Market benchmark OOS
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

            # Regime distribution
            regime_dist = oos['regime'].value_counts(normalize=True)

            record = {
                'Pair': pair_name,
                'Sector': row['Sector'],
                'Discovery_Score': row['Score'],
                'Coint_pval': row['Coint_pval'],
                'Half_Life': row['Half_Life'],
                # Training performance
                'Train_Return_%': train_total * 100,
                'Train_Sharpe': train_sharpe,
                # OOS performance
                'OOS_Return_%': oos_total * 100,
                'OOS_Ann_Return_%': oos_ann * 100,
                'OOS_Sharpe': oos_sharpe,
                'OOS_Vol_%': oos_vol * 100,
                'OOS_MaxDD_%': max_dd * 100,
                'OOS_Trades': n_trades,
                'OOS_WinRate_%': win_rate * 100,
                'OOS_AvgPnL_%': avg_pnl,
                'Market_Return_%': mkt_total * 100,
                'Alpha_%': (oos_total - mkt_total) * 100,
                'OOS_Days': len(oos_ret),
                'Crisis_%': regime_dist.get(0, 0) * 100,
            }

            # Monte Carlo p-value (quick version with fewer sims)
            if run_monte_carlo and n_trades >= 3:
                mc = MonteCarloValidator(
                    n_simulations=1000, random_seed=42 + idx
                )
                mc_result = mc.run(system, period='oos', verbose=False)
                record['MC_pvalue'] = mc_result['p_value_sharpe']
            else:
                record['MC_pvalue'] = np.nan

            # Bootstrap CI
            if run_bootstrap and len(oos_ret) >= 30:
                bs = BootstrapAnalyzer(n_resamples=1000, random_seed=42 + idx)
                bs_result = bs.run(oos_ret, verbose=False)
                record['BS_Sharpe_Lo'] = bs_result['sharpe_ci_lower']
                record['BS_Sharpe_Hi'] = bs_result['sharpe_ci_upper']
            else:
                record['BS_Sharpe_Lo'] = np.nan
                record['BS_Sharpe_Hi'] = np.nan

            results.append(record)

            if verbose:
                mc_str = ""
                if not np.isnan(record['MC_pvalue']):
                    mc_str = f" MC-p={record['MC_pvalue']:.3f}"
                print(f"    ✓ OOS: Ret={oos_total*100:+.2f}% "
                      f"Sharpe={oos_sharpe:.2f} "
                      f"Trades={n_trades} "
                      f"WR={win_rate*100:.0f}%{mc_str}")

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            continue

        # Rate limit
        time.sleep(0.5)

    return pd.DataFrame(results)


# =============================================================================
# AGGREGATE ANALYSIS
# =============================================================================

def aggregate_analysis(results_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Compute aggregate statistics across the universe."""
    if len(results_df) == 0:
        return {}

    n = len(results_df)
    profitable = results_df[results_df['OOS_Return_%'] > 0]
    positive_alpha = results_df[results_df['Alpha_%'] > 0]

    agg = {
        'Total Pairs Tested': n,
        'Profitable Pairs': len(profitable),
        'Hit Rate (% profitable)': len(profitable) / n * 100,
        'Positive Alpha Pairs': len(positive_alpha),
        'Alpha Hit Rate (%)': len(positive_alpha) / n * 100,
        'Median OOS Return (%)': results_df['OOS_Return_%'].median(),
        'Mean OOS Return (%)': results_df['OOS_Return_%'].mean(),
        'Median OOS Sharpe': results_df['OOS_Sharpe'].median(),
        'Mean OOS Sharpe': results_df['OOS_Sharpe'].mean(),
        'Best Pair': results_df.loc[results_df['OOS_Sharpe'].idxmax(), 'Pair'],
        'Best Sharpe': results_df['OOS_Sharpe'].max(),
        'Worst Pair': results_df.loc[results_df['OOS_Sharpe'].idxmin(), 'Pair'],
        'Worst Sharpe': results_df['OOS_Sharpe'].min(),
        'Median Max DD (%)': results_df['OOS_MaxDD_%'].median(),
        'Avg OOS Trades': results_df['OOS_Trades'].mean(),
        'Avg Win Rate (%)': results_df['OOS_WinRate_%'].mean(),
    }

    # Sector breakdown
    sector_stats = results_df.groupby('Sector').agg({
        'OOS_Sharpe': ['mean', 'count'],
        'OOS_Return_%': 'mean',
        'Alpha_%': 'mean',
    }).round(3)
    agg['Sector Breakdown'] = sector_stats

    # Train vs OOS degradation
    train_sharpe = results_df['Train_Sharpe'].mean()
    oos_sharpe = results_df['OOS_Sharpe'].mean()
    agg['Train→OOS Sharpe Drop'] = f"{train_sharpe:.3f} → {oos_sharpe:.3f}"
    agg['Overfit Ratio'] = train_sharpe / oos_sharpe if oos_sharpe != 0 else float('inf')

    # BH correction on Monte Carlo p-values
    mc_pvals = results_df[['Pair', 'MC_pvalue']].dropna()
    if len(mc_pvals) > 0:
        bh_results = benjamini_hochberg(
            list(zip(mc_pvals['Pair'], mc_pvals['MC_pvalue'])),
            alpha=0.05
        )
        n_sig_raw = sum(1 for _, p, _, _ in bh_results if p < 0.05)
        n_sig_bh = sum(1 for _, _, _, sig in bh_results if sig)
        agg['MC Significant (raw p<0.05)'] = n_sig_raw
        agg['MC Significant (BH corrected)'] = n_sig_bh
        agg['BH Details'] = bh_results
    else:
        agg['MC Significant (raw p<0.05)'] = 0
        agg['MC Significant (BH corrected)'] = 0

    if verbose:
        print("\n" + "=" * 70)
        print("  UNIVERSE AGGREGATE STATISTICS")
        print("=" * 70)
        for k, v in agg.items():
            if k in ('Sector Breakdown', 'BH Details'):
                continue
            print(f"  {k:.<45} {v}")

        print("\n  SECTOR BREAKDOWN:")
        print(sector_stats.to_string())

        if 'BH Details' in agg:
            print("\n  BENJAMINI-HOCHBERG CORRECTION:")
            for name, raw_p, adj_p, sig in agg['BH Details']:
                icon = "✅" if sig else "  "
                print(f"    {icon} {name:.<25} raw={raw_p:.4f}  adj={adj_p:.4f}")

    return agg


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_universe_report(results_df: pd.DataFrame,
                             agg: dict, save_dir: str = None) -> str:
    """Generate and save the full universe backtest report."""
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(save_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(save_dir, 'universe_summary.csv')
    results_df.to_csv(csv_path, index=False)

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("UNIVERSE BACKTEST REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    lines.append("KEY METRICS")
    lines.append("-" * 80)
    for k, v in agg.items():
        if k in ('Sector Breakdown', 'BH Details'):
            continue
        lines.append(f"  {k:.<50} {v}")
    lines.append("")

    lines.append("TOP 10 PAIRS BY OOS SHARPE")
    lines.append("-" * 80)
    top = results_df.nlargest(10, 'OOS_Sharpe')
    for _, row in top.iterrows():
        lines.append(
            f"  {row['Pair']:.<15} Sharpe={row['OOS_Sharpe']:.3f}  "
            f"Ret={row['OOS_Return_%']:+.2f}%  "
            f"Alpha={row['Alpha_%']:+.2f}%  "
            f"Trades={int(row['OOS_Trades'])}  "
            f"WR={row['OOS_WinRate_%']:.0f}%"
        )
    lines.append("")

    lines.append("BOTTOM 5 PAIRS BY OOS SHARPE")
    lines.append("-" * 80)
    bot = results_df.nsmallest(5, 'OOS_Sharpe')
    for _, row in bot.iterrows():
        lines.append(
            f"  {row['Pair']:.<15} Sharpe={row['OOS_Sharpe']:.3f}  "
            f"Ret={row['OOS_Return_%']:+.2f}%"
        )
    lines.append("")

    # Overall assessment
    lines.append("=" * 80)
    lines.append("OVERALL ASSESSMENT")
    lines.append("=" * 80)
    hit_rate = agg.get('Hit Rate (% profitable)', 0)
    median_sharpe = agg.get('Median OOS Sharpe', 0)
    bh_sig = agg.get('MC Significant (BH corrected)', 0)
    total = agg.get('Total Pairs Tested', 1)

    if hit_rate > 60 and median_sharpe > 0.3:
        lines.append("✅ STRONG: Strategy works across multiple pairs")
        lines.append(f"   {hit_rate:.0f}% profitable, median Sharpe {median_sharpe:.3f}")
    elif hit_rate > 50 and median_sharpe > 0:
        lines.append("⚠️ MODERATE: Strategy has merit but not universal")
    else:
        lines.append("❌ WEAK: Strategy fails across most pairs")

    if bh_sig > 0:
        lines.append(f"   {bh_sig}/{total} pairs survive BH correction")

    report_text = "\n".join(lines)

    report_path = os.path.join(save_dir, 'universe_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n  Reports saved:")
    print(f"    CSV:  {csv_path}")
    print(f"    Text: {report_path}")

    return report_text


# =============================================================================
# MAIN
# =============================================================================

def run_universe_backtest(min_score: int = 40,
                          max_pairs_per_sector: int = 10,
                          start_date: str = '2015-01-01',
                          end_date: str = '2025-06-30',
                          train_end_date: str = '2020-12-31',
                          slippage_bps: float = 5.0,
                          run_mc: bool = True,
                          run_bs: bool = True) -> tuple:
    """Full pipeline: discover → backtest → aggregate → report."""
    cfg = load_config()

    print("=" * 70)
    print("  UNIVERSE BACKTEST — FULL PIPELINE")
    print(f"  {start_date} → {end_date} | Train ≤ {train_end_date}")
    print(f"  Min score: {min_score} | Slippage: {slippage_bps}bps")
    print("=" * 70)

    # Step 1: Discover pairs
    print("\n📡 STEP 1: PAIR DISCOVERY")
    pairs = discover_pairs(
        start_date=start_date, end_date=end_date,
        min_score=min_score,
        max_pairs_per_sector=max_pairs_per_sector,
    )

    if len(pairs) == 0:
        print("  ✗ No viable pairs found!")
        return pd.DataFrame(), {}

    # Save discovered pairs
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    pairs.to_csv(os.path.join(results_dir, 'discovered_pairs.csv'), index=False)

    # Step 2: Backtest all pairs
    print(f"\n📊 STEP 2: BACKTESTING {len(pairs)} PAIRS")
    bt_results = backtest_universe(
        pairs, start_date=start_date, end_date=end_date,
        train_end_date=train_end_date, slippage_bps=slippage_bps,
        run_monte_carlo=run_mc, run_bootstrap=run_bs,
    )

    if len(bt_results) == 0:
        print("  ✗ All backtests failed!")
        return pd.DataFrame(), {}

    # Step 3: Aggregate
    print(f"\n📈 STEP 3: AGGREGATE ANALYSIS")
    agg = aggregate_analysis(bt_results)

    # Step 4: Report
    print(f"\n📝 STEP 4: GENERATING REPORT")
    generate_universe_report(bt_results, agg)

    return bt_results, agg


if __name__ == "__main__":
    results, aggregate = run_universe_backtest(
        min_score=40,
        max_pairs_per_sector=10,
        start_date='2015-01-01',
        end_date='2025-06-30',
        train_end_date='2020-12-31',
        slippage_bps=5.0,
        run_mc=True,
        run_bs=True,
    )
    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(results)} pairs backtested")
    print(f"{'=' * 70}")
