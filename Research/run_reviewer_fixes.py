"""
Reviewer Fix Script — All 10 Improvements for 8.5/10 Paper
===========================================================
Addresses ALL computational issues identified in the IEEE Access review:

  Fix 1: Bootstrap/MC for success pairs (corrected API calls)
  Fix 2: RL detailed reporting (Flat%, holding period, 4-decimal Sharpe)
  Fix 3: Lo(2002) analytical Sharpe CIs for all pairs
  Fix 4: Economic significance (annualised $, capacity, break-even cost)
  Fix 5: Walk-forward conditional metrics (when healthy, avg fold Sharpe)

Usage:
    cd /Users/hardik/PBL_Run
    source venv/bin/activate
    python Research/run_reviewer_fixes.py          # full mode
    python Research/run_reviewer_fixes.py --quick   # fast validation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from Core_Strategy.strategy_validator import (
    MonteCarloValidator, BootstrapAnalyzer,
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
VAL_END   = '2022-12-31'
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


# =========================================================================
# UTILITY: Lo (2002) analytical Sharpe ratio SE
# =========================================================================
def lo_sharpe_se(sharpe, n_obs):
    """
    Standard error of Sharpe ratio using Lo (2002) formula:
      SE(SR) ≈ sqrt((1 + 0.5 * SR^2) / n)
    where SR is annualised Sharpe and n = number of daily observations.
    """
    return np.sqrt((1 + 0.5 * sharpe ** 2) / n_obs)


def lo_sharpe_ci(sharpe, n_obs, alpha=0.05):
    """95% CI for Sharpe ratio using Lo (2002)."""
    se = lo_sharpe_se(sharpe, n_obs)
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    return sharpe - z * se, sharpe + z * se


# =========================================================================
# FIX 1: BOOTSTRAP & MONTE CARLO FOR SUCCESS PAIRS
# =========================================================================
def fix_bootstrap_mc(quick=False):
    """
    Re-run bootstrap (correct API: BootstrapAnalyzer(n_resamples=...).run())
    and Monte Carlo (direct implementation since MC.run() requires ConservativeSystem).

    For MC: we implement a direct random-entry MC test on the filtered returns.
    """
    print("\n" + "=" * 70)
    print("  FIX 1: BOOTSTRAP & MONTE CARLO FOR SUCCESS PAIRS")
    print("=" * 70)

    success_pairs = [('AEP', 'SRE'), ('CVX', 'COP')]
    n_mc = 1000 if quick else 10000
    n_boot = 1000 if quick else 10000
    results_all = {}

    for ty, tx in success_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── {pair_name} ──")

        with timer(f'bootstrap_mc_{ty}_{tx}'):
            stock_y, stock_x, market = download_data(
                ty, tx, 'SPY', '2015-01-01', '2025-06-30'
            )

            result = backtest_with_dynamic_filter(
                stock_y, stock_x, market,
                train_end_date=TRAIN_END,
                health_step=5, verbose=False, slippage_bps=5.0,
            )

            # Extract filtered OOS returns
            filtered_bt = result['filtered']
            oos_mask = filtered_bt['period'] == 'oos'
            oos_ret = filtered_bt.loc[oos_mask, 'strategy_return'].dropna()

            mf = result['metrics_filtered']['oos']
            actual_sharpe = mf['sharpe']
            n_obs = len(oos_ret)

            print(f"    Filtered Sharpe: {actual_sharpe:.4f}, N obs: {n_obs}")

            # ── Bootstrap (correct API) ──
            print(f"    Running block bootstrap ({n_boot} resamples)...")
            try:
                ba = BootstrapAnalyzer(n_resamples=n_boot, confidence_level=0.95)
                boot_result = ba.run(oos_ret, verbose=False)
                boot_ci_lo = boot_result['sharpe_ci_lower']
                boot_ci_hi = boot_result['sharpe_ci_upper']
                boot_se = boot_result['sharpe_se']
                print(f"    Bootstrap 95% CI: [{boot_ci_lo:.3f}, {boot_ci_hi:.3f}]")
                print(f"    Bootstrap SE: {boot_se:.3f}")
            except Exception as e:
                print(f"    Bootstrap still failed: {e}")
                traceback.print_exc()
                boot_ci_lo, boot_ci_hi, boot_se = 0, 0, 0

            # ── Lo (2002) analytical CI ──
            lo_ci_lo, lo_ci_hi = lo_sharpe_ci(actual_sharpe, n_obs)
            lo_se = lo_sharpe_se(actual_sharpe, n_obs)
            print(f"    Lo(2002) 95% CI: [{lo_ci_lo:.3f}, {lo_ci_hi:.3f}]")
            print(f"    Lo(2002) SE: {lo_se:.3f}")

            # ── Monte Carlo (direct implementation) ──
            # Since success pairs have very sparse trading (1.8% - 6.4% healthy),
            # we implement a direct random-entry MC that respects the sparsity.
            print(f"    Running Monte Carlo ({n_mc} random strategies)...")

            spread_ret = filtered_bt.loc[oos_mask, 'spread_return'].dropna().values
            actual_signal = filtered_bt.loc[oos_mask, 'final_signal'].fillna(0).values
            n_days = len(spread_ret)

            # Count actual trade characteristics
            sig_changes = np.diff(actual_signal)
            n_trades_actual = int(np.sum(np.abs(sig_changes) > 0)) // 2
            in_position = np.abs(actual_signal) > 0
            pct_in_position = in_position.mean()

            # Compute actual strategy return
            actual_strat_ret = np.roll(actual_signal, 1) * spread_ret
            actual_strat_ret[0] = 0
            actual_total = np.prod(1 + actual_strat_ret) - 1
            if np.std(actual_strat_ret) > 0:
                actual_ann = (1 + actual_total) ** (252 / n_days) - 1
                actual_vol = np.std(actual_strat_ret) * np.sqrt(252)
                actual_mc_sharpe = actual_ann / actual_vol
            else:
                actual_mc_sharpe = 0

            rng = np.random.RandomState(42)
            sim_sharpes = np.zeros(n_mc)

            for s in range(n_mc):
                # Random entry strategy with same activity level
                sim_signal = np.zeros(n_days)
                i = 0
                trades_placed = 0

                while trades_placed < max(n_trades_actual, 3) and i < n_days - 2:
                    gap = rng.geometric(p=max(n_trades_actual, 3) / n_days)
                    i += gap
                    if i >= n_days - 2:
                        break

                    hold = max(2, int(rng.exponential(max(5, pct_in_position * n_days / max(n_trades_actual, 1)))))
                    hold = min(hold, n_days - i - 1)

                    direction = rng.choice([-1, 1])
                    sim_signal[i:i + hold] = direction
                    i += hold
                    trades_placed += 1

                sim_ret = np.roll(sim_signal, 1) * spread_ret
                sim_ret[0] = 0
                total = np.prod(1 + sim_ret) - 1
                vol = np.std(sim_ret) * np.sqrt(252)
                if vol > 0 and total > -1:
                    ann = (1 + total) ** (252 / n_days) - 1
                    sim_sharpes[s] = ann / vol
                else:
                    sim_sharpes[s] = 0

            mc_pvalue = (sim_sharpes >= actual_mc_sharpe).mean()
            mc_mean = sim_sharpes.mean()
            mc_std = sim_sharpes.std()

            print(f"    MC p-value: {mc_pvalue:.4f}")
            print(f"    MC random Sharpe: {mc_mean:.3f} ± {mc_std:.3f}")
            print(f"    Actual trades: {n_trades_actual}, in-position: {pct_in_position*100:.1f}%")

            results_all[pair_name] = {
                'actual_sharpe': actual_sharpe,
                'n_obs': n_obs,
                'boot_ci_lo': boot_ci_lo,
                'boot_ci_hi': boot_ci_hi,
                'boot_se': boot_se,
                'lo_ci_lo': lo_ci_lo,
                'lo_ci_hi': lo_ci_hi,
                'lo_se': lo_se,
                'mc_pvalue': mc_pvalue,
                'mc_mean': mc_mean,
                'mc_std': mc_std,
                'n_trades': n_trades_actual,
                'pct_in_position': pct_in_position,
            }

    # Save
    df = pd.DataFrame(results_all).T
    df.index.name = 'Pair'
    df.to_csv(os.path.join(RESULTS_DIR, 'success_pairs_significance.csv'))

    with open(os.path.join(RESULTS_DIR, 'success_pairs_significance.json'), 'w') as f:
        json.dump(results_all, f, indent=2, default=float)

    print(f"\n  ✓ Saved success_pairs_significance.csv/json")
    return results_all


# =========================================================================
# FIX 2: RL DETAILED REPORTING
# =========================================================================
def fix_rl_reporting(quick=False):
    """
    Re-run RL with detailed diagnostics: Flat%, avg holding period,
    actual Sharpe to 4 decimals, signal distribution, and action analysis.
    """
    print("\n" + "=" * 70)
    print("  FIX 2: RL DETAILED REPORTING")
    print("=" * 70)

    from Research.rl_pairs_agent import (
        run_rl_backtest, AgentConfig, PairsTradingEnv, EnvConfig,
    )

    rl_pairs = [
        ('BAC', 'PNC'),
        ('WFC', 'MS'),
        ('CVX', 'OXY'),
        ('AEP', 'SRE'),
    ]

    max_eps = 30 if quick else 150
    all_results = []

    for ty, tx in rl_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── RL Agent: {pair_name} ──")

        with timer(f'rl_detailed_{ty}_{tx}'):
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
                hist = result['training_history']

                # Detailed signal analysis
                rl_signals = result['signals_rl']
                n_total = len(rl_signals)

                # OOS portion only
                # Find OOS start from backtest data
                bt_data = result.get('bt_data', None)
                if bt_data is not None and 'period' in bt_data.columns:
                    oos_index = bt_data.index[bt_data['period'] == 'oos']
                    rl_oos = rl_signals.reindex(oos_index).fillna(0)
                else:
                    # Approximate: last ~1126 days
                    train_end_loc = pd.Timestamp(TRAIN_END)
                    if hasattr(rl_signals, 'index') and hasattr(rl_signals.index, 'get_loc'):
                        try:
                            split_idx = rl_signals.index.get_indexer([train_end_loc], method='ffill')[0]
                            rl_oos = rl_signals.iloc[split_idx:]
                        except:
                            rl_oos = rl_signals.iloc[len(rl_signals)//2:]
                    else:
                        rl_oos = rl_signals.iloc[len(rl_signals)//2:]

                n_oos = len(rl_oos)
                n_flat = int((rl_oos == 0).sum())
                n_long = int((rl_oos > 0).sum())
                n_short = int((rl_oos < 0).sum())
                flat_pct = n_flat / n_oos * 100 if n_oos > 0 else 100

                # Compute actual holding periods
                sig_diff = rl_oos.diff().fillna(0)
                entries = sig_diff != 0
                if entries.sum() > 0:
                    # Compute runs of non-zero signals
                    in_pos = (rl_oos != 0).astype(int)
                    runs = []
                    current_run = 0
                    for v in in_pos.values:
                        if v > 0:
                            current_run += 1
                        else:
                            if current_run > 0:
                                runs.append(current_run)
                            current_run = 0
                    if current_run > 0:
                        runs.append(current_run)
                    avg_hold = np.mean(runs) if runs else 0
                    n_round_trips = len(runs)
                else:
                    avg_hold = 0
                    n_round_trips = 0

                # Compute actual (non-rounded) Sharpe
                rl_sharpe_raw = rl_m['sharpe']
                rl_return_raw = rl_m['total_return']
                rl_maxdd_raw = rl_m['max_dd']

                print(f"    Baseline Sharpe: {bl_m['sharpe']:.4f}")
                print(f"    RL OOS Sharpe:   {rl_sharpe_raw:.4f}")
                print(f"    RL OOS Return:   {rl_return_raw:.6f}")
                print(f"    RL OOS MaxDD:    {rl_maxdd_raw:.6f}")
                print(f"    Flat %:          {flat_pct:.1f}%  ({n_flat}/{n_oos} days)")
                print(f"    Long days:       {n_long}  |  Short days: {n_short}")
                print(f"    Round-trip trades: {n_round_trips}")
                print(f"    Avg holding:     {avg_hold:.1f} days")
                print(f"    Val Sharpe:      {hist['best_val_sharpe']:.4f}")

                all_results.append({
                    'Pair': pair_name,
                    'BL_Sharpe': round(bl_m['sharpe'], 4),
                    'RL_Sharpe': round(rl_sharpe_raw, 4),
                    'Sharpe_Delta': round(rl_sharpe_raw - bl_m['sharpe'], 4),
                    'Val_Sharpe': round(hist['best_val_sharpe'], 4),
                    'RL_Trades': n_round_trips,
                    'Flat_Pct': round(flat_pct, 1),
                    'Long_Days': n_long,
                    'Short_Days': n_short,
                    'Avg_Hold_Days': round(avg_hold, 1),
                    'RL_Return': round(rl_return_raw, 6),
                    'RL_MaxDD': round(rl_maxdd_raw, 6),
                    'BL_Trades': bl_m['n_trades'],
                    'Episodes': hist['total_episodes'],
                })

            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                all_results.append({
                    'Pair': pair_name,
                    'BL_Sharpe': 0, 'RL_Sharpe': 0, 'Sharpe_Delta': 0,
                    'Val_Sharpe': 0, 'RL_Trades': 0, 'Flat_Pct': 100,
                    'Long_Days': 0, 'Short_Days': 0, 'Avg_Hold_Days': 0,
                    'RL_Return': 0, 'RL_MaxDD': 0, 'BL_Trades': 0, 'Episodes': 0,
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, 'rl_detailed.csv'), index=False)
    print(f"\n  RL Detailed Summary:")
    print(df.to_string(index=False))
    print(f"\n  ✓ Saved rl_detailed.csv")

    return df


# =========================================================================
# FIX 3: Lo(2002) SHARPE CIs + ECONOMIC SIGNIFICANCE FOR ALL PAIRS
# =========================================================================
def compute_economic_significance(quick=False):
    """
    For all key pairs: compute Lo(2002) Sharpe CIs, annualised dollar return,
    capacity estimate, and break-even transaction cost.
    """
    print("\n" + "=" * 70)
    print("  FIX 3: ECONOMIC SIGNIFICANCE & ANALYTICAL SHARPE CIs")
    print("=" * 70)

    all_pairs = [
        ('BAC', 'PNC', 'Banking'),
        ('WFC', 'MS', 'Financials'),
        ('CVX', 'OXY', 'Energy'),
        ('PNC', 'CFG', 'Banking'),
        ('FCX', 'VMC', 'Materials'),
    ]

    capital = 100_000  # $100K per pair
    results = []

    for ty, tx, sector in all_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── {pair_name} ({sector}) ──")

        with timer(f'econ_{ty}_{tx}'):
            stock_y, stock_x, market = download_data(
                ty, tx, 'SPY', '2015-01-01', '2025-06-30'
            )

            result = backtest_with_dynamic_filter(
                stock_y, stock_x, market,
                train_end_date=TRAIN_END,
                health_step=5, verbose=False, slippage_bps=5.0,
            )

            mu = result['metrics_unfiltered']['oos']
            mf = result['metrics_filtered']['oos']
            fs = result['filter_stats']

            # Filtered returns
            filtered_bt = result['filtered']
            oos_mask = filtered_bt['period'] == 'oos'
            oos_ret = filtered_bt.loc[oos_mask, 'strategy_return'].dropna()
            n_obs = len(oos_ret)
            n_years = n_obs / 252

            # --- Lo(2002) CI for both unfiltered and filtered ---
            f_sharpe = mf['sharpe']
            uf_sharpe = mu['sharpe']

            f_lo_lo, f_lo_hi = lo_sharpe_ci(f_sharpe, n_obs)
            uf_lo_lo, uf_lo_hi = lo_sharpe_ci(uf_sharpe, n_obs)

            # --- Economic significance ---
            f_total_ret = mf['total_return']
            f_ann_ret = (1 + f_total_ret) ** (1 / n_years) - 1 if f_total_ret > -1 else -1
            f_dollar_pnl = capital * f_total_ret
            f_ann_dollar = capital * f_ann_ret

            # Capacity estimate: based on market impact model
            # Assume pair has ~$50M combined daily volume
            # Market impact = 0.1 * sigma * sqrt(V/ADV)
            # At $100K per trade with $50M ADV, impact is negligible
            # Estimate max capacity where market impact costs < 50% of returns
            avg_daily_ret = oos_ret.mean()
            if avg_daily_ret > 0:
                # Max capital where 0.1 * vol * sqrt(C/ADV) < 0.5 * avg_daily_return
                # Approximate: C_max ≈ (0.5 * avg_ret / (0.1 * vol))^2 * ADV
                daily_vol = oos_ret.std()
                adv_estimate = 50e6  # conservative
                if daily_vol > 0:
                    ratio = (0.5 * avg_daily_ret / (0.1 * daily_vol))
                    capacity = min(ratio ** 2 * adv_estimate, 100e6)
                else:
                    capacity = 0
            else:
                capacity = 0

            # Break-even cost analysis
            # Current cost: 5 bps slippage + $1 commission
            # Find cost at which Sharpe = 0
            # Run backtest with different cost levels
            cost_levels = [0, 2, 5, 10, 15, 20, 30]
            sharpe_at_cost = {}
            for cost_bps in cost_levels:
                try:
                    r = backtest_with_dynamic_filter(
                        stock_y, stock_x, market,
                        train_end_date=TRAIN_END,
                        health_step=5, verbose=False, slippage_bps=float(cost_bps),
                    )
                    sharpe_at_cost[cost_bps] = r['metrics_filtered']['oos']['sharpe']
                except:
                    sharpe_at_cost[cost_bps] = np.nan

            # Interpolate break-even cost
            costs = sorted(sharpe_at_cost.keys())
            sharpes = [sharpe_at_cost[c] for c in costs]
            breakeven_cost = None
            for i in range(len(costs) - 1):
                if sharpes[i] > 0 and sharpes[i+1] <= 0:
                    # Linear interpolation
                    frac = sharpes[i] / (sharpes[i] - sharpes[i+1])
                    breakeven_cost = costs[i] + frac * (costs[i+1] - costs[i])
                    break
            if breakeven_cost is None and all(s > 0 for s in sharpes if not np.isnan(s)):
                breakeven_cost = max(costs)  # still profitable at max cost

            print(f"    Filtered Sharpe: {f_sharpe:.4f}")
            print(f"    Lo(2002) 95% CI: [{f_lo_lo:.3f}, {f_lo_hi:.3f}]")
            ci_str = "CI excludes 0 ✅" if f_lo_lo > 0 else "CI includes 0"
            print(f"    {ci_str}")
            print(f"    Ann. dollar return on $100K: ${f_ann_dollar:,.0f}")
            print(f"    Total PnL over OOS: ${f_dollar_pnl:,.0f}")
            print(f"    Est. capacity: ${capacity/1e6:.1f}M")
            print(f"    Break-even cost: {breakeven_cost:.1f} bps" if breakeven_cost else "    Break-even cost: N/A (never profitable)")
            print(f"    Sharpe at costs: {sharpe_at_cost}")

            results.append({
                'Pair': pair_name,
                'Sector': sector,
                'F_Sharpe': round(f_sharpe, 4),
                'UF_Sharpe': round(uf_sharpe, 4),
                'F_Lo_CI_Lo': round(f_lo_lo, 3),
                'F_Lo_CI_Hi': round(f_lo_hi, 3),
                'UF_Lo_CI_Lo': round(uf_lo_lo, 3),
                'UF_Lo_CI_Hi': round(uf_lo_hi, 3),
                'CI_Excludes_Zero': f_lo_lo > 0,
                'N_OOS_Days': n_obs,
                'F_Ann_Dollar_100K': round(f_ann_dollar, 0),
                'F_Total_PnL_100K': round(f_dollar_pnl, 0),
                'Capacity_Est_M': round(capacity / 1e6, 1) if capacity > 0 else 0,
                'Breakeven_Cost_Bps': round(breakeven_cost, 1) if breakeven_cost else None,
                'Healthy_Pct': round(fs['oos_pct_healthy'], 1),
                'Sharpe_at_0bps': sharpe_at_cost.get(0, np.nan),
                'Sharpe_at_5bps': sharpe_at_cost.get(5, np.nan),
                'Sharpe_at_10bps': sharpe_at_cost.get(10, np.nan),
                'Sharpe_at_20bps': sharpe_at_cost.get(20, np.nan),
                'Sharpe_at_30bps': sharpe_at_cost.get(30, np.nan),
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'economic_significance.csv'), index=False)

    print(f"\n  Economic Significance Summary:")
    print(df[['Pair', 'Sector', 'F_Sharpe', 'F_Lo_CI_Lo', 'F_Lo_CI_Hi',
              'CI_Excludes_Zero', 'F_Ann_Dollar_100K', 'Breakeven_Cost_Bps']].to_string(index=False))
    print(f"\n  ✓ Saved economic_significance.csv")

    # Generate break-even cost figure
    generate_breakeven_figure(results)

    return df


def generate_breakeven_figure(results):
    """fig18: Break-even cost analysis for all 5 pairs."""
    print(f"\n  Generating fig18_breakeven_cost...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Sharpe vs transaction cost
    cost_cols = [0, 5, 10, 20, 30]
    for r in results:
        sharpes = [r.get(f'Sharpe_at_{c}bps', np.nan) for c in cost_cols]
        style = '-o' if r['F_Sharpe'] > 0 else '--s'
        ax1.plot(cost_cols, sharpes, style, label=r['Pair'], markersize=5, linewidth=1.5)

    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Transaction Cost (bps)')
    ax1.set_ylabel('Filtered OOS Sharpe')
    ax1.set_title('Sharpe vs. Transaction Cost\n(Break-Even Analysis)', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel B: Dollar PnL per $100K capital
    pairs = [r['Pair'] for r in results]
    pnls = [r['F_Ann_Dollar_100K'] for r in results]
    colors = ['#1976d2' if p > 0 else '#d32f2f' for p in pnls]

    bars = ax2.bar(pairs, pnls, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Annualised Return ($) per $100K')
    ax2.set_title('Economic Significance\n(Annual Dollar Returns)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)

    for bar, pnl in zip(bars, pnls):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${pnl:,.0f}', ha='center', va='bottom' if pnl > 0 else 'top',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig18_breakeven_cost.{ext}'))
    plt.close(fig)
    print(f"    ✓ Saved fig18_breakeven_cost.pdf/png")


# =========================================================================
# FIX 4: WALK-FORWARD CONDITIONAL METRICS
# =========================================================================
def compute_wf_conditional(quick=False):
    """
    For success pairs: compute conditional WF metrics
    (when healthy, average fold Sharpe; magnitude of positive vs negative folds).
    """
    print("\n" + "=" * 70)
    print("  FIX 4: WALK-FORWARD CONDITIONAL METRICS")
    print("=" * 70)

    pairs = [('AEP', 'SRE'), ('CVX', 'COP')]
    n_folds = 8 if quick else 32
    min_train = 504
    fold_size = 126
    step_size = 63

    results = {}

    for ty, tx in pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── {pair_name} walk-forward conditional analysis ──")

        stock_y, stock_x, market = download_data(
            ty, tx, 'SPY', '2015-01-01', '2025-06-30'
        )

        system = ConservativeSystem()
        n_days = len(stock_y)
        dates = stock_y.index

        fold_data = []

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

                # Health check for this fold's test window
                monitor = PairHealthMonitor()
                health = monitor.compute_rolling_health(
                    stock_y, stock_x, bt['spread'], step=5
                )

                test_dates = dates[test_start_idx:min(test_end_idx, len(dates))]
                test_health = health.reindex(test_dates, method='ffill')
                if 'is_healthy' in test_health.columns:
                    pct_healthy = test_health['is_healthy'].mean() * 100
                else:
                    pct_healthy = 0

                # Filtered returns
                from Research.dynamic_pair_selector import apply_health_gate
                filtered_sig = apply_health_gate(bt['final_signal'], health, grace_period=5)
                test_mask = (bt.index >= test_dates[0]) & (bt.index <= test_dates[-1])
                test_ret = (filtered_sig.shift(1) * bt['spread_return'])[test_mask].dropna()

                if len(test_ret) > 10 and test_ret.std() > 0:
                    fold_sharpe = (test_ret.mean() / test_ret.std()) * np.sqrt(252)
                    fold_return = test_ret.sum()
                else:
                    fold_sharpe = 0.0
                    fold_return = 0.0

                fold_data.append({
                    'fold': fold,
                    'sharpe': fold_sharpe,
                    'return': fold_return,
                    'pct_healthy': pct_healthy,
                    'n_days': len(test_ret),
                    'is_positive': fold_sharpe > 0,
                })

            except Exception as e:
                fold_data.append({
                    'fold': fold, 'sharpe': 0, 'return': 0,
                    'pct_healthy': 0, 'n_days': 0, 'is_positive': False,
                })

        fd = pd.DataFrame(fold_data)
        pos_folds = fd[fd['sharpe'] > 0]
        neg_folds = fd[fd['sharpe'] <= 0]

        # Conditional metrics
        avg_pos_sharpe = pos_folds['sharpe'].mean() if len(pos_folds) > 0 else 0
        avg_neg_sharpe = neg_folds['sharpe'].mean() if len(neg_folds) > 0 else 0
        avg_pos_return = pos_folds['return'].mean() if len(pos_folds) > 0 else 0
        avg_neg_return = neg_folds['return'].mean() if len(neg_folds) > 0 else 0

        # When healthy folds (>5% healthy) vs unhealthy
        healthy_folds = fd[fd['pct_healthy'] > 5]
        unhealthy_folds = fd[fd['pct_healthy'] <= 5]
        avg_healthy_sharpe = healthy_folds['sharpe'].mean() if len(healthy_folds) > 0 else 0
        avg_unhealthy_sharpe = unhealthy_folds['sharpe'].mean() if len(unhealthy_folds) > 0 else 0

        results[pair_name] = {
            'n_folds': len(fd),
            'n_positive': int(fd['is_positive'].sum()),
            'pct_positive': fd['is_positive'].mean() * 100,
            'avg_all_sharpe': fd['sharpe'].mean(),
            'avg_pos_sharpe': avg_pos_sharpe,
            'avg_neg_sharpe': avg_neg_sharpe,
            'avg_pos_return': avg_pos_return,
            'avg_neg_return': avg_neg_return,
            'pos_neg_magnitude_ratio': abs(avg_pos_sharpe / avg_neg_sharpe) if avg_neg_sharpe != 0 else float('inf'),
            'n_healthy_folds': len(healthy_folds),
            'n_unhealthy_folds': len(unhealthy_folds),
            'avg_healthy_sharpe': avg_healthy_sharpe,
            'avg_unhealthy_sharpe': avg_unhealthy_sharpe,
            'fold_sharpes': fd['sharpe'].tolist(),
        }

        print(f"    Total folds: {len(fd)}")
        print(f"    Positive: {int(fd['is_positive'].sum())}/{len(fd)} ({fd['is_positive'].mean()*100:.1f}%)")
        print(f"    Avg positive fold Sharpe: {avg_pos_sharpe:.3f}")
        print(f"    Avg negative fold Sharpe: {avg_neg_sharpe:.3f}")
        print(f"    Magnitude ratio (pos/neg): {results[pair_name]['pos_neg_magnitude_ratio']:.2f}")
        print(f"    Healthy folds ({len(healthy_folds)}): avg Sharpe {avg_healthy_sharpe:.3f}")
        print(f"    Unhealthy folds ({len(unhealthy_folds)}): avg Sharpe {avg_unhealthy_sharpe:.3f}")

    with open(os.path.join(RESULTS_DIR, 'wf_conditional.json'), 'w') as f:
        # Remove non-serializable fold_sharpes for JSON
        save_results = {}
        for k, v in results.items():
            save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'fold_sharpes'}
        json.dump(save_results, f, indent=2, default=float)

    print(f"\n  ✓ Saved wf_conditional.json")
    return results


# =========================================================================
# FIX 5: Lo(2002) CIs FOR ALL 12 WILCOXON PAIRS
# =========================================================================
def compute_all_lo_cis():
    """Compute Lo(2002) Sharpe CIs for all 12 pairs used in Wilcoxon test."""
    print("\n" + "=" * 70)
    print("  FIX 5: Lo(2002) SHARPE CIs FOR ALL 12 PAIRS")
    print("=" * 70)

    test_pairs = [
        ('BAC', 'PNC'), ('WFC', 'MS'), ('CVX', 'OXY'),
        ('AEP', 'SRE'), ('CVX', 'COP'),
        ('VMC', 'MLM'), ('LIN', 'VMC'),
        ('JNJ', 'UNH'), ('GOOGL', 'AMD'),
        ('KO', 'COST'), ('HON', 'DE'), ('USB', 'TFC'),
    ]

    results = []
    for ty, tx in test_pairs:
        pair_name = f"{ty}/{tx}"
        print(f"  {pair_name}...", end=" ")

        try:
            stock_y, stock_x, market = download_data(
                ty, tx, 'SPY', '2015-01-01', '2025-06-30'
            )
            result = backtest_with_dynamic_filter(
                stock_y, stock_x, market,
                train_end_date=TRAIN_END,
                health_step=5, verbose=False, slippage_bps=5.0,
            )

            mf = result['metrics_filtered']['oos']
            mu = result['metrics_unfiltered']['oos']
            filtered_bt = result['filtered']
            oos_ret = filtered_bt[filtered_bt['period'] == 'oos']['strategy_return'].dropna()
            n = len(oos_ret)

            f_lo, f_hi = lo_sharpe_ci(mf['sharpe'], n)
            uf_lo, uf_hi = lo_sharpe_ci(mu['sharpe'], n)

            results.append({
                'Pair': pair_name,
                'F_Sharpe': round(mf['sharpe'], 4),
                'F_Lo_CI': f"[{f_lo:.3f}, {f_hi:.3f}]",
                'F_CI_Excl_Zero': f_lo > 0,
                'UF_Sharpe': round(mu['sharpe'], 4),
                'UF_Lo_CI': f"[{uf_lo:.3f}, {uf_hi:.3f}]",
                'N': n,
            })
            ci_mark = "✅" if f_lo > 0 else "○"
            print(f"F={mf['sharpe']:.3f} CI=[{f_lo:.3f},{f_hi:.3f}] {ci_mark}")

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({'Pair': pair_name, 'F_Sharpe': 0, 'F_Lo_CI': 'N/A',
                          'F_CI_Excl_Zero': False, 'UF_Sharpe': 0, 'UF_Lo_CI': 'N/A', 'N': 0})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'lo_sharpe_cis.csv'), index=False)
    print(f"\n  ✓ Saved lo_sharpe_cis.csv")
    n_excl = df['F_CI_Excl_Zero'].sum()
    print(f"  {n_excl}/{len(df)} pairs have filtered Sharpe CI excluding zero")
    return df


###############################################################################
# MAIN
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Reviewer fix experiments')
    parser.add_argument('--quick', action='store_true', help='Fast mode')
    args = parser.parse_args()
    quick = args.quick

    print(f"\n{'#'*70}")
    print(f"  REVIEWER FIX EXPERIMENTS — {'QUICK' if quick else 'FULL'} MODE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    t0 = time.time()

    # Fix 1: Bootstrap/MC for success pairs
    sig_results = fix_bootstrap_mc(quick=quick)

    # Fix 2: RL detailed reporting
    rl_df = fix_rl_reporting(quick=quick)

    # Fix 3: Economic significance + analytical CIs for 5 key pairs
    econ_df = compute_economic_significance(quick=quick)

    # Fix 4: Walk-forward conditional metrics
    wf_cond = compute_wf_conditional(quick=quick)

    # Fix 5: Lo(2002) CIs for all 12 pairs
    lo_df = compute_all_lo_cis()

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'#'*70}")
    print(f"  ALL REVIEWER FIXES COMPLETE — {elapsed/60:.1f} min total")
    print(f"{'#'*70}")

    print(f"\n  New files in Research/results/:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f:<45} {size:>8,} bytes")

    print(f"\n  TIMING:")
    for k, v in sorted(TIMING.items()):
        print(f"    {k:<35} {v:.1f}s")


if __name__ == '__main__':
    main()
