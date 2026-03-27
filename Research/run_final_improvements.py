"""
Final Paper Improvements — Boost from 7.5 to 8.5/10
======================================================
Runs ALL computational experiments for the 5 remaining reviewer items:

  1. Full (non-quick) backtest on AEP/SRE & CVX/COP with walk-forward + bootstrap
  2. Wilcoxon signed-rank test on 10-pair filter DD reduction
  3. RL on WFC/MS, CVX/OXY, and AEP/SRE (expanded pair)
  4. Generate new figures + tables
  5. Collect rotation system data for paper integration

Usage:
    cd /Users/hardik/PBL_Run
    source venv/bin/activate
    python Research/run_final_improvements.py
    python Research/run_final_improvements.py --quick   # faster subset
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
import time
import json
import warnings
import traceback
from datetime import datetime
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, download_data, calculate_half_life,
)
from Research.dynamic_pair_selector import (
    PairHealthMonitor, HealthThresholds, backtest_with_dynamic_filter,
)
from Research.rl_pairs_agent import (
    PairsTradingEnv, DQNetwork, DQNAgent, AgentConfig, EnvConfig,
)

# ── Paths ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'Research', 'results')
FIGURES_DIR = os.path.join(ROOT, 'Paper', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Plot style ──
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.figsize': (8, 5),
})

TRAIN_END = '2020-12-31'
VAL_END = '2022-12-31'

TIMING = {}

def timer(name):
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *_):
            elapsed = time.time() - self.start
            TIMING[name] = elapsed
            print(f"  ⏱  {name}: {elapsed:.1f}s")
    return Timer()


###############################################################################
# 1. FULL BACKTEST ON SUCCESS PAIRS: AEP/SRE & CVX/COP
###############################################################################

def run_success_pair_backtests(quick=False):
    """
    Run full filtered + unfiltered backtests on the two best-performing pairs.
    Include walk-forward analysis and bootstrap confidence intervals.
    """
    print("\n" + "=" * 70)
    print("  ITEM 1: FULL BACKTEST ON SUCCESS PAIRS (AEP/SRE, CVX/COP)")
    print("=" * 70)

    from Core_Strategy.strategy_validator import (
        MonteCarloValidator, BootstrapAnalyzer, benjamini_hochberg
    )

    success_pairs = [('AEP', 'SRE'), ('CVX', 'COP')]
    results = {}

    for ty, tx in success_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── {pair_name} ──")

        with timer(f'full_backtest_{ty}_{tx}'):
            stock_y, stock_x, market = download_data(
                ty, tx, 'SPY', '2015-01-01', '2025-06-30'
            )

            # Full filtered backtest
            print(f"    Running filtered backtest...")
            result = backtest_with_dynamic_filter(
                stock_y, stock_x, market,
                train_end_date=TRAIN_END,
                health_step=5,
                verbose=False,
                slippage_bps=5.0,
            )
            results[pair_name] = result

            mu = result['metrics_unfiltered']['oos']
            mf = result['metrics_filtered']['oos']
            fs = result['filter_stats']

            print(f"    Unfiltered: Sharpe={mu['sharpe']:.3f}, Return={mu['total_return']*100:.1f}%, "
                  f"MaxDD={mu['max_dd']*100:.1f}%")
            print(f"    Filtered:   Sharpe={mf['sharpe']:.3f}, Return={mf['total_return']*100:.1f}%, "
                  f"MaxDD={mf['max_dd']*100:.1f}%")
            print(f"    Healthy: {fs['oos_pct_healthy']:.1f}%")

        # Walk-Forward Analysis
        print(f"    Running walk-forward (32 folds)...")
        with timer(f'walkforward_{ty}_{tx}'):
            wf_results = run_walk_forward(stock_y, stock_x, market, pair_name, quick=quick)

        # Bootstrap CI
        if not quick:
            print(f"    Running bootstrap CI (10,000 resamples)...")
            with timer(f'bootstrap_{ty}_{tx}'):
                filtered_bt = result['filtered']
                oos_ret = filtered_bt[filtered_bt['period'] == 'oos']['strategy_return'].dropna()
                try:
                    ba = BootstrapAnalyzer(n_bootstrap=10000, block_size=5)
                    bootstrap_result = ba.bootstrap_sharpe(oos_ret)
                    ci_low = bootstrap_result.get('ci_lower', 0)
                    ci_high = bootstrap_result.get('ci_upper', 0)
                    print(f"    Bootstrap Sharpe 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
                    results[pair_name]['bootstrap_ci'] = (ci_low, ci_high)
                except Exception as e:
                    print(f"    Bootstrap failed: {e}")
                    results[pair_name]['bootstrap_ci'] = (0, 0)

            # Monte Carlo
            print(f"    Running Monte Carlo (10,000 trials)...")
            with timer(f'montecarlo_{ty}_{tx}'):
                try:
                    mcv = MonteCarloValidator(n_simulations=10000)
                    signals = filtered_bt[filtered_bt['period'] == 'oos']['final_signal']
                    spread_ret = filtered_bt[filtered_bt['period'] == 'oos']['spread_return']
                    mc_result = mcv.run_full_monte_carlo(
                        signals, spread_ret,
                        actual_sharpe=mf['sharpe'],
                        regime_mask=None,
                    )
                    mc_pval = mc_result.get('p_value', 1.0)
                    print(f"    MC p-value: {mc_pval:.4f}")
                    results[pair_name]['mc_pvalue'] = mc_pval
                except Exception as e:
                    print(f"    MC failed: {e}")
                    results[pair_name]['mc_pvalue'] = 1.0
        else:
            results[pair_name]['bootstrap_ci'] = (0, 0)
            results[pair_name]['mc_pvalue'] = 1.0

        results[pair_name]['walk_forward'] = wf_results

    # Save results
    summary_rows = []
    for pair_name, r in results.items():
        mu = r['metrics_unfiltered']['oos']
        mf = r['metrics_filtered']['oos']
        fs = r['filter_stats']
        bs = r.get('bootstrap_ci', (0, 0))
        mc = r.get('mc_pvalue', 1.0)
        wf = r.get('walk_forward', {})

        summary_rows.append({
            'Pair': pair_name,
            'UF_Sharpe': mu['sharpe'],
            'UF_Return': mu['total_return'],
            'UF_MaxDD': mu['max_dd'],
            'F_Sharpe': mf['sharpe'],
            'F_Return': mf['total_return'],
            'F_MaxDD': mf['max_dd'],
            'Pct_Healthy': fs['oos_pct_healthy'],
            'Bootstrap_CI_Low': bs[0],
            'Bootstrap_CI_High': bs[1],
            'MC_pvalue': mc,
            'WF_Mean_Sharpe': wf.get('mean_sharpe', 0),
            'WF_Pct_Positive': wf.get('pct_positive', 0),
            'WF_N_Folds': wf.get('n_folds', 0),
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'success_pairs_full.csv'), index=False)
    print(f"\n  ✓ Saved success_pairs_full.csv")

    # Generate equity curve figure (fig15)
    generate_success_pairs_figure(results)

    return results


def run_walk_forward(stock_y, stock_x, market, pair_name, quick=False):
    """Walk-forward with 32 expanding folds on filtered backtest."""
    n_folds = 8 if quick else 32
    min_train = 504
    fold_size = 126
    step_size = 63

    system = ConservativeSystem()
    n_days = len(stock_y)
    fold_sharpes = []

    dates = stock_y.index
    for fold in range(n_folds):
        train_end_idx = min_train + fold * step_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_days)

        if test_end_idx > n_days or test_start_idx >= n_days:
            break

        train_end_date = dates[train_end_idx - 1].strftime('%Y-%m-%d')

        try:
            bt = system.run_backtest(
                stock_y, stock_x, market,
                train_end_date=train_end_date,
                verbose=False, slippage_bps=5.0,
            )

            # Apply health filter
            monitor = PairHealthMonitor()
            health = monitor.compute_rolling_health(
                stock_y, stock_x, bt['spread'], step=5
            )
            from Research.dynamic_pair_selector import apply_health_gate
            filtered_sig = apply_health_gate(bt['final_signal'], health, grace_period=5)

            # Compute filtered returns for test window
            test_mask = (bt.index >= dates[test_start_idx]) & (bt.index < dates[min(test_end_idx, len(dates)-1)])
            test_ret = (filtered_sig.shift(1) * bt['spread_return'])[test_mask].dropna()

            if len(test_ret) > 10 and test_ret.std() > 0:
                fold_sharpe = (test_ret.mean() / test_ret.std()) * np.sqrt(252)
            else:
                fold_sharpe = 0.0

            fold_sharpes.append(fold_sharpe)
        except Exception:
            fold_sharpes.append(0.0)

    if fold_sharpes:
        return {
            'fold_sharpes': fold_sharpes,
            'mean_sharpe': np.mean(fold_sharpes),
            'median_sharpe': np.median(fold_sharpes),
            'pct_positive': sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes) * 100,
            'n_folds': len(fold_sharpes),
        }
    return {'fold_sharpes': [], 'mean_sharpe': 0, 'median_sharpe': 0, 'pct_positive': 0, 'n_folds': 0}


def generate_success_pairs_figure(results):
    """Generate fig15: Success pair equity curves with CI bands."""
    print(f"\n  Generating fig15_success_pairs...")

    n_pairs = len(results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5), sharey=False)
    if n_pairs == 1:
        axes = [axes]

    for idx, (pair_name, result) in enumerate(results.items()):
        ax = axes[idx]
        bt = result['unfiltered']
        ft = result['filtered']

        oos_mask = bt['period'] == 'oos'
        uf_cum = (1 + bt.loc[oos_mask, 'strategy_return']).cumprod()
        ft_cum = (1 + ft.loc[oos_mask, 'strategy_return']).cumprod()

        ax.plot(uf_cum.index, uf_cum.values, color='#d32f2f',
                linewidth=1.5, label='Unfiltered', alpha=0.7)
        ax.plot(ft_cum.index, ft_cum.values, color='#1976d2',
                linewidth=2.0, label='Filtered', alpha=0.9)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # Shade unhealthy regions
        health = result['health']
        oos_health = health.reindex(bt.loc[oos_mask].index, method='ffill')
        if 'is_healthy' in oos_health.columns:
            unhealthy = ~oos_health['is_healthy'].fillna(True)
            for i in range(len(unhealthy)):
                if unhealthy.iloc[i]:
                    ax.axvspan(unhealthy.index[i],
                              unhealthy.index[min(i+1, len(unhealthy)-1)],
                              alpha=0.06, color='red')

        mu = result['metrics_unfiltered']['oos']
        mf = result['metrics_filtered']['oos']

        ax.set_title(f"{pair_name}\nFiltered Sharpe: {mf['sharpe']:+.3f} | "
                     f"DD: {mf['max_dd']*100:.1f}%", fontweight='bold', fontsize=11)
        ax.set_ylabel('Cumulative Return' if idx == 0 else '')
        ax.legend(loc='best', fontsize=9)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Success Pairs: Health-Filtered OOS Performance',
                 fontweight='bold', y=1.02, fontsize=13)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig15_success_pairs.{ext}'))
    plt.close(fig)
    print(f"    ✓ Saved fig15_success_pairs.pdf/png")


###############################################################################
# 2. WILCOXON SIGNED-RANK TEST ON FILTER IMPROVEMENT
###############################################################################

def run_wilcoxon_test(quick=False):
    """
    Formal significance test on the dynamic filter's DD reduction.
    Uses 10+ pairs from the expanded universe.
    Computes Wilcoxon signed-rank on paired (unfiltered DD, filtered DD).
    """
    print("\n" + "=" * 70)
    print("  ITEM 2: WILCOXON SIGNED-RANK TEST ON FILTER DD IMPROVEMENT")
    print("=" * 70)

    # All pairs to test (expanded universe)
    test_pairs = [
        ('BAC', 'PNC'), ('WFC', 'MS'), ('CVX', 'OXY'),
        ('AEP', 'SRE'), ('CVX', 'COP'),
        ('VMC', 'MLM'), ('LIN', 'VMC'),
        ('JNJ', 'UNH'), ('GOOGL', 'AMD'),
        ('KO', 'COST'), ('HON', 'DE'), ('USB', 'TFC'),
    ]

    if quick:
        test_pairs = test_pairs[:6]

    pair_stats = []

    for ty, tx in test_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  Testing {pair_name}...", end=" ")

        try:
            with timer(f'wilcoxon_{ty}_{tx}'):
                stock_y, stock_x, market = download_data(
                    ty, tx, 'SPY', '2015-01-01', '2025-06-30'
                )
                result = backtest_with_dynamic_filter(
                    stock_y, stock_x, market,
                    train_end_date=TRAIN_END,
                    health_step=5,
                    verbose=False,
                    slippage_bps=5.0,
                )

                mu = result['metrics_unfiltered']['oos']
                mf = result['metrics_filtered']['oos']
                fs = result['filter_stats']

                pair_stats.append({
                    'Pair': pair_name,
                    'UF_Sharpe': mu['sharpe'],
                    'F_Sharpe': mf['sharpe'],
                    'Sharpe_Delta': mf['sharpe'] - mu['sharpe'],
                    'UF_MaxDD': mu['max_dd'],
                    'F_MaxDD': mf['max_dd'],
                    'DD_Improvement': abs(mf['max_dd']) - abs(mu['max_dd']),  # negative = filter helped
                    'DD_Reduction_Pct': (abs(mu['max_dd']) - abs(mf['max_dd'])) / max(abs(mu['max_dd']), 0.001) * 100,
                    'UF_Return': mu['total_return'],
                    'F_Return': mf['total_return'],
                    'Pct_Healthy': fs['oos_pct_healthy'],
                })
                print(f"UF_DD={mu['max_dd']*100:.1f}% → F_DD={mf['max_dd']*100:.1f}%")

        except Exception as e:
            print(f"FAILED: {e}")

    df = pd.DataFrame(pair_stats)

    # Wilcoxon signed-rank test on MaxDD (absolute values, paired)
    uf_dd = np.abs(df['UF_MaxDD'].values)
    f_dd = np.abs(df['F_MaxDD'].values)

    # Test: is filtered DD significantly less than unfiltered DD?
    stat_dd, p_dd = scipy_stats.wilcoxon(uf_dd, f_dd, alternative='greater')
    print(f"\n  {'='*60}")
    print(f"  WILCOXON SIGNED-RANK: MaxDD Reduction")
    print(f"  {'='*60}")
    print(f"  N pairs: {len(df)}")
    print(f"  Median UF |MaxDD|: {np.median(uf_dd)*100:.1f}%")
    print(f"  Median F  |MaxDD|: {np.median(f_dd)*100:.1f}%")
    print(f"  Wilcoxon statistic: {stat_dd:.1f}")
    print(f"  p-value (one-sided): {p_dd:.6f}")
    print(f"  Significant at α=0.05: {'YES ✅' if p_dd < 0.05 else 'NO ❌'}")
    print(f"  Significant at α=0.01: {'YES ✅' if p_dd < 0.01 else 'NO ❌'}")

    # Also test Sharpe improvement
    uf_sh = df['UF_Sharpe'].values
    f_sh = df['F_Sharpe'].values
    stat_sh, p_sh = scipy_stats.wilcoxon(f_sh - uf_sh, alternative='greater')
    print(f"\n  WILCOXON SIGNED-RANK: Sharpe Improvement")
    print(f"  Median UF Sharpe: {np.median(uf_sh):.3f}")
    print(f"  Median F  Sharpe: {np.median(f_sh):.3f}")
    print(f"  Wilcoxon statistic: {stat_sh:.1f}")
    print(f"  p-value (one-sided): {p_sh:.6f}")
    print(f"  Significant at α=0.05: {'YES ✅' if p_sh < 0.05 else 'NO ❌'}")

    # Summary table
    df['Filter_Helped_DD'] = df['DD_Reduction_Pct'] > 0
    df['Filter_Helped_Sharpe'] = df['Sharpe_Delta'] > 0

    print(f"\n  Filter helped DD for {df['Filter_Helped_DD'].sum()}/{len(df)} pairs")
    print(f"  Filter helped Sharpe for {df['Filter_Helped_Sharpe'].sum()}/{len(df)} pairs")

    # Save
    df.to_csv(os.path.join(RESULTS_DIR, 'wilcoxon_filter_test.csv'), index=False)

    test_summary = {
        'n_pairs': len(df),
        'dd_wilcoxon_stat': float(stat_dd),
        'dd_pvalue': float(p_dd),
        'dd_significant_005': bool(p_dd < 0.05),
        'sharpe_wilcoxon_stat': float(stat_sh),
        'sharpe_pvalue': float(p_sh),
        'sharpe_significant_005': bool(p_sh < 0.05),
        'median_uf_dd': float(np.median(uf_dd)),
        'median_f_dd': float(np.median(f_dd)),
        'median_uf_sharpe': float(np.median(uf_sh)),
        'median_f_sharpe': float(np.median(f_sh)),
        'pairs_dd_helped': int(df['Filter_Helped_DD'].sum()),
        'pairs_sharpe_helped': int(df['Filter_Helped_Sharpe'].sum()),
    }
    with open(os.path.join(RESULTS_DIR, 'wilcoxon_summary.json'), 'w') as f:
        json.dump(test_summary, f, indent=2)

    print(f"\n  ✓ Saved wilcoxon_filter_test.csv + wilcoxon_summary.json")

    # Generate figure
    generate_wilcoxon_figure(df, test_summary)

    return df, test_summary


def generate_wilcoxon_figure(df, summary):
    """fig16: Paired DD comparison with Wilcoxon test annotation."""
    print(f"\n  Generating fig16_wilcoxon_filter...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Paired DD bar chart
    pairs = df['Pair'].values
    x = np.arange(len(pairs))
    width = 0.35

    uf_dd = np.abs(df['UF_MaxDD'].values) * 100
    f_dd = np.abs(df['F_MaxDD'].values) * 100

    bars1 = ax1.bar(x - width/2, uf_dd, width, label='Unfiltered', color='#d32f2f', alpha=0.8)
    bars2 = ax1.bar(x + width/2, f_dd, width, label='Filtered', color='#1976d2', alpha=0.8)

    ax1.set_xlabel('Pair')
    ax1.set_ylabel('|Max Drawdown| (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    p_dd = summary['dd_pvalue']
    sig_str = f"p = {p_dd:.4f}" if p_dd >= 0.0001 else f"p < 0.0001"
    ax1.set_title(f'Max Drawdown Reduction\nWilcoxon {sig_str} {"✱✱" if p_dd < 0.01 else "✱" if p_dd < 0.05 else "n.s."}',
                  fontweight='bold')

    # Panel B: Sharpe improvement
    sharpe_delta = df['Sharpe_Delta'].values
    colors = ['#1976d2' if d > 0 else '#d32f2f' for d in sharpe_delta]
    ax2.barh(x, sharpe_delta, color=colors, alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(pairs, fontsize=8)
    ax2.set_xlabel('Sharpe Δ (Filtered − Unfiltered)')
    ax2.axvline(0, color='black', linewidth=0.8)
    p_sh = summary['sharpe_pvalue']
    sig_str_sh = f"p = {p_sh:.4f}" if p_sh >= 0.0001 else f"p < 0.0001"
    ax2.set_title(f'Sharpe Improvement\nWilcoxon {sig_str_sh}',
                  fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig16_wilcoxon_filter.{ext}'))
    plt.close(fig)
    print(f"    ✓ Saved fig16_wilcoxon_filter.pdf/png")


###############################################################################
# 3. RL ON MULTIPLE PAIRS (WFC/MS, CVX/OXY, AEP/SRE)
###############################################################################

def run_rl_multi_pair(quick=False):
    """
    Run DQN agent on WFC/MS, CVX/OXY, and AEP/SRE to generalize the
    RL finding beyond just BAC/PNC.
    Uses the existing run_rl_backtest() pipeline for each pair.
    """
    print("\n" + "=" * 70)
    print("  ITEM 3: RL ON MULTIPLE PAIRS")
    print("=" * 70)

    from Research.rl_pairs_agent import run_rl_backtest

    rl_pairs = [
        ('BAC', 'PNC'),    # Re-run for consistency
        ('WFC', 'MS'),     # Core pair
        ('CVX', 'OXY'),    # Core pair
        ('AEP', 'SRE'),    # Expanded success pair
    ]

    max_eps = 30 if quick else 150
    all_results = []

    for ty, tx in rl_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── RL Agent: {pair_name} ──")

        with timer(f'rl_{ty}_{tx}'):
            try:
                result = run_rl_backtest(
                    ticker_y=ty, ticker_x=tx,
                    start_date='2015-01-01', end_date='2025-06-30',
                    train_end_date=TRAIN_END, val_end_date=VAL_END,
                    slippage_bps=5.0,
                    agent_config=AgentConfig(
                        max_episodes=max_eps,
                        epsilon_decay=0.995 if not quick else 0.99,
                        early_stop_patience=20 if not quick else 10,
                    ),
                    verbose=False,
                )

                rl_m = result['rl_metrics']
                bl_m = result['baseline_metrics']
                comp = result['comparison']
                hist = result['training_history']

                all_results.append({
                    'Pair': pair_name,
                    'Baseline_Sharpe': bl_m['sharpe'],
                    'RL_Sharpe': rl_m['sharpe'],
                    'Sharpe_Delta': comp['sharpe_delta'],
                    'RL_Trades': rl_m['n_trades'],
                    'Baseline_Trades': bl_m['n_trades'],
                    'RL_Return': rl_m['total_return'],
                    'BL_Return': bl_m['total_return'],
                    'RL_MaxDD': rl_m['max_dd'],
                    'BL_MaxDD': bl_m['max_dd'],
                    'Val_Sharpe': hist['best_val_sharpe'],
                    'Episodes': hist['total_episodes'],
                    'RL_Improved': comp['rl_improved'],
                })

                print(f"    Baseline Sharpe: {bl_m['sharpe']:.3f}")
                print(f"    RL OOS Sharpe:   {rl_m['sharpe']:.3f} (delta: {comp['sharpe_delta']:+.3f})")
                print(f"    RL Trades:       {rl_m['n_trades']} | Val Sharpe: {hist['best_val_sharpe']:.3f}")

            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                all_results.append({
                    'Pair': pair_name, 'Baseline_Sharpe': 0, 'RL_Sharpe': 0,
                    'Sharpe_Delta': 0, 'RL_Trades': 0, 'Baseline_Trades': 0,
                    'RL_Return': 0, 'BL_Return': 0, 'RL_MaxDD': 0, 'BL_MaxDD': 0,
                    'Val_Sharpe': 0, 'Episodes': 0, 'RL_Improved': False,
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, 'rl_multi_pair.csv'), index=False)
    print(f"\n  RL Multi-Pair Summary:")
    print(df.to_string(index=False))
    print(f"\n  ✓ Saved rl_multi_pair.csv")

    return df


###############################################################################
# 4. ROTATION SYSTEM DATA FOR PAPER
###############################################################################

def collect_rotation_data():
    """
    Collect the 38-pair rotation scan results for paper integration.
    Also reads the saved rotation state.
    """
    print("\n" + "=" * 70)
    print("  ITEM 4: ROTATION SYSTEM DATA")
    print("=" * 70)

    from Paper_Trading.pair_rotation import PairRotationManager, EXPANDED_UNIVERSE

    manager = PairRotationManager(top_k=3)

    print(f"\n  Universe: {len(EXPANDED_UNIVERSE)} pairs across "
          f"{len(set(v['sector'] for v in EXPANDED_UNIVERSE.values()))} sectors")

    # Run scan
    scan = manager.scan_universe(verbose=True)
    rankings = manager.rank_pairs(scan)
    selected = manager.select_top_pairs(rankings, scan)

    # Build summary table
    rotation_rows = []
    for rank, (name, score, eligible) in enumerate(rankings, 1):
        info = scan.get(name, {})
        rotation_rows.append({
            'Rank': rank,
            'Pair': name,
            'Sector': info.get('sector', '?'),
            'Health_Score': score,
            'Is_Healthy': info.get('is_healthy', False),
            'Eligible': eligible,
            'Selected': name in selected,
            'Recent_Healthy_Pct': info.get('pct_healthy_recent', 0),
            'ADF_pval': info.get('adf_pvalue', np.nan),
            'Hurst': info.get('hurst', np.nan),
            'Coint_pval': info.get('coint_pvalue', np.nan),
        })

    df = pd.DataFrame(rotation_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'rotation_38pair_scan.csv'), index=False)

    print(f"\n  Top-3 Selected: {', '.join(selected)}")
    print(f"  Total eligible: {sum(1 for _, _, e in rankings if e)}")
    print(f"  Total scanned: {len(rankings)}")

    # Load rotation state
    state_file = os.path.join(ROOT, 'Paper_Trading', 'logs', 'rotation_state.json')
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        print(f"\n  Current rotation state:")
        print(f"    Active pairs: {state.get('current_pairs', [])}")
        print(f"    Last rotation: {state.get('last_rotation_date', 'N/A')}")

    # Generate rotation figure
    generate_rotation_figure(df)

    print(f"\n  ✓ Saved rotation_38pair_scan.csv")
    return df


def generate_rotation_figure(df):
    """fig17: 38-pair health ranking bar chart with selection cutoff."""
    print(f"\n  Generating fig17_rotation_rankings...")

    fig, ax = plt.subplots(figsize=(14, 6))

    pairs = df['Pair'].values
    scores = df['Health_Score'].values
    selected = df['Selected'].values
    eligible = df['Eligible'].values

    colors = []
    for sel, elig in zip(selected, eligible):
        if sel:
            colors.append('#1976d2')   # Selected — blue
        elif elig:
            colors.append('#43a047')   # Eligible but not selected — green
        else:
            colors.append('#bdbdbd')   # Not eligible — grey

    bars = ax.bar(range(len(pairs)), scores, color=colors, edgecolor='white', linewidth=0.5)

    # Add threshold line
    ax.axhline(0.50, color='#d32f2f', linestyle='--', linewidth=1.2,
               label='Eligibility threshold (0.50)')

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Health Score')
    ax.set_title('38-Pair Universe Health Scan — Monthly Rotation Rankings',
                 fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1976d2', label=f'Selected (top-3)'),
        Patch(facecolor='#43a047', label='Eligible'),
        Patch(facecolor='#bdbdbd', label='Excluded'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig17_rotation_rankings.{ext}'))
    plt.close(fig)
    print(f"    ✓ Saved fig17_rotation_rankings.pdf/png")


###############################################################################
# MAIN
###############################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Final paper improvements')
    parser.add_argument('--quick', action='store_true', help='Fast mode')
    args = parser.parse_args()

    quick = args.quick
    print(f"\n{'#'*70}")
    print(f"  FINAL PAPER IMPROVEMENTS — {'QUICK' if quick else 'FULL'} MODE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    t0 = time.time()

    # 1. Success pair full backtests
    success_results = run_success_pair_backtests(quick=quick)

    # 2. Wilcoxon test
    wilcoxon_df, wilcoxon_summary = run_wilcoxon_test(quick=quick)

    # 3. RL multi-pair
    rl_df = run_rl_multi_pair(quick=quick)

    # 4. Rotation data
    rotation_df = collect_rotation_data()

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  ALL IMPROVEMENTS COMPLETE — {elapsed/60:.1f} min total")
    print(f"{'#'*70}")

    print(f"\n  Results saved to Research/results/:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    {f}")

    print(f"\n  Figures saved to Paper/figures/:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.startswith('fig1') and (f.endswith('.pdf') or f.endswith('.png')):
            print(f"    {f}")

    print(f"\n  TIMING:")
    for k, v in sorted(TIMING.items()):
        print(f"    {k:<35} {v:.1f}s")

    # Save all timing
    with open(os.path.join(RESULTS_DIR, 'final_timing.json'), 'w') as f:
        json.dump(TIMING, f, indent=2)


if __name__ == '__main__':
    main()
