"""
Day-5 full-scale success-pair validation.

Select success pairs from expanded universe results and run the full
pipeline (filtered/unfiltered backtest, walk-forward, bootstrap, MC).
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy import stats as scipy_stats
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import ConservativeSystem, download_data
from Core_Strategy.strategy_validator import BootstrapAnalyzer
from Research.dynamic_pair_selector import (
    PairHealthMonitor,
    backtest_with_dynamic_filter,
    apply_health_gate,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'Research', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'Paper', 'figures')

TRAIN_END = '2020-12-31'
OOS_DAYS = 1126


def lo_sharpe_ci(sharpe, n_obs, alpha=0.05):
    """Lo (2002) Sharpe CI with IID approximation."""
    if n_obs <= 1:
        return 0.0, 0.0
    se = math.sqrt((1 + 0.5 * (sharpe ** 2)) / max(n_obs, 1))
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    return sharpe - z * se, sharpe + z * se


def select_success_pairs(csv_path, min_sharpe=0.0, min_healthy_pct=10.0,
                         min_eff_days=300, min_pairs=5):
    df = pd.read_csv(csv_path)
    df['Effective_Days'] = (df['Pct_Healthy'] / 100.0) * OOS_DAYS

    base = df[(df['F_OOS_Sharpe'] > min_sharpe) &
              ((df['Pct_Healthy'] >= min_healthy_pct) | (df['Effective_Days'] >= min_eff_days))]
    reason = f"Sharpe>0 & Pct_Healthy>={min_healthy_pct}"

    if len(base) < min_pairs:
        relaxed = df[(df['F_OOS_Sharpe'] > min_sharpe) & (df['Pct_Healthy'] >= 7.0)]
        if len(relaxed) >= min_pairs:
            base = relaxed
            reason = "Relaxed: Sharpe>0 & Pct_Healthy>=7"

    if len(base) < min_pairs:
        base = df[df['F_OOS_Sharpe'] > min_sharpe].sort_values(
            'F_OOS_Sharpe', ascending=False
        ).head(min_pairs)
        reason = "Fallback: top Sharpe>0"

    selection = base[['Pair', 'Sector', 'F_OOS_Sharpe', 'Pct_Healthy', 'Effective_Days']].copy()
    selection['Reason'] = reason

    selection_path = os.path.join(RESULTS_DIR, 'success_pairs_selection.csv')
    selection.to_csv(selection_path, index=False)
    print(f"  ✓ Saved {selection_path}")

    pairs = []
    for pair in base['Pair']:
        ty, tx = pair.split('_')
        pairs.append((ty, tx))

    return pairs


def run_walk_forward(stock_y, stock_x, market, quick=False):
    """Walk-forward with filtered signals on each fold."""
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

            monitor = PairHealthMonitor()
            health = monitor.compute_rolling_health(
                stock_y, stock_x, bt['spread'], step=5
            )
            filtered_sig = apply_health_gate(bt['final_signal'], health, grace_period=5)

            test_mask = (bt.index >= dates[test_start_idx]) & (bt.index < dates[min(test_end_idx, len(dates) - 1)])
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
            'mean_sharpe': float(np.mean(fold_sharpes)),
            'median_sharpe': float(np.median(fold_sharpes)),
            'pct_positive': sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes) * 100,
            'n_folds': len(fold_sharpes),
        }
    return {'fold_sharpes': [], 'mean_sharpe': 0, 'median_sharpe': 0, 'pct_positive': 0, 'n_folds': 0}


def monte_carlo_sparse(filtered_bt, n_mc=10000, seed=42):
    """Random-entry MC for sparse filtered signals."""
    oos_mask = filtered_bt['period'] == 'oos'
    spread_ret = filtered_bt.loc[oos_mask, 'spread_return'].dropna().values
    actual_signal = filtered_bt.loc[oos_mask, 'final_signal'].fillna(0).values
    n_days = len(spread_ret)
    if n_days == 0:
        return 1.0

    sig_changes = np.diff(actual_signal)
    n_trades_actual = int(np.sum(np.abs(sig_changes) > 0)) // 2
    in_position = np.abs(actual_signal) > 0
    pct_in_position = in_position.mean()

    actual_strat_ret = np.roll(actual_signal, 1) * spread_ret
    actual_strat_ret[0] = 0
    actual_total = np.prod(1 + actual_strat_ret) - 1
    if np.std(actual_strat_ret) > 0:
        actual_ann = (1 + actual_total) ** (252 / n_days) - 1
        actual_vol = np.std(actual_strat_ret) * np.sqrt(252)
        actual_mc_sharpe = actual_ann / actual_vol
    else:
        actual_mc_sharpe = 0

    rng = np.random.RandomState(seed)
    sim_sharpes = np.zeros(n_mc)

    for s in range(n_mc):
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

    return float((sim_sharpes >= actual_mc_sharpe).mean())


def generate_success_pairs_figure(results):
    """Generate fig15: Success pair equity curves with CI bands."""
    print("\n  Generating fig15_success_pairs...")

    n_pairs = len(results)
    n_cols = 2 if n_pairs <= 4 else 3
    n_rows = int(math.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), sharey=False)
    axes = np.array(axes).reshape(-1)

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

        health = result['health']
        oos_health = health.reindex(bt.loc[oos_mask].index, method='ffill')
        if 'is_healthy' in oos_health.columns:
            unhealthy = ~oos_health['is_healthy'].fillna(True)
            for i in range(len(unhealthy)):
                if unhealthy.iloc[i]:
                    ax.axvspan(unhealthy.index[i],
                              unhealthy.index[min(i + 1, len(unhealthy) - 1)],
                              alpha=0.06, color='red')

        mf = result['metrics_filtered']['oos']

        ax.set_title(f"{pair_name}\nFiltered Sharpe: {mf['sharpe']:+.3f} | "
                     f"DD: {mf['max_dd']*100:.1f}%", fontweight='bold', fontsize=10)
        ax.set_ylabel('Cumulative Return' if idx % n_cols == 0 else '')
        ax.legend(loc='best', fontsize=8)
        ax.tick_params(axis='x', rotation=30)

    for j in range(n_pairs, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Success Pairs: Health-Filtered OOS Performance',
                 fontweight='bold', y=1.02, fontsize=13)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig15_success_pairs.{ext}'))
    plt.close(fig)
    print("    ✓ Saved fig15_success_pairs.pdf/png")


def run_full_pipeline(pairs, quick=False):
    from matplotlib import pyplot as plt
    globals()['plt'] = plt

    n_mc = 1000 if quick else 10000
    n_boot = 1000 if quick else 10000

    results = {}
    for ty, tx in pairs:
        pair_name = f"{ty}/{tx}"
        print(f"\n  ── {pair_name} ──")

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

        filtered_bt = result['filtered']
        oos_mask = filtered_bt['period'] == 'oos'
        oos_ret = filtered_bt.loc[oos_mask, 'strategy_return'].dropna()
        n_obs = len(oos_ret)

        mf = result['metrics_filtered']['oos']

        ba = BootstrapAnalyzer(n_resamples=n_boot, confidence_level=0.95)
        boot = ba.run(oos_ret, verbose=False)

        lo_ci_lo, lo_ci_hi = lo_sharpe_ci(mf['sharpe'], n_obs)
        mc_pvalue = monte_carlo_sparse(filtered_bt, n_mc=n_mc)

        wf = run_walk_forward(stock_y, stock_x, market, quick=quick)

        result['bootstrap_ci'] = (boot['sharpe_ci_lower'], boot['sharpe_ci_upper'])
        result['lo_ci'] = (lo_ci_lo, lo_ci_hi)
        result['mc_pvalue'] = mc_pvalue
        result['walk_forward'] = wf
        result['n_obs'] = n_obs

        results[pair_name] = result

        print(f"    Filtered Sharpe: {mf['sharpe']:+.3f} | OOS n={n_obs}")
        print(f"    Lo CI: [{lo_ci_lo:+.3f}, {lo_ci_hi:+.3f}] | MC p={mc_pvalue:.4f}")
        print(f"    Bootstrap CI: [{boot['sharpe_ci_lower']:+.3f}, {boot['sharpe_ci_upper']:+.3f}]")

    summary_rows = []
    for pair_name, r in results.items():
        mu = r['metrics_unfiltered']['oos']
        mf = r['metrics_filtered']['oos']
        fs = r['filter_stats']
        bs = r.get('bootstrap_ci', (0, 0))
        lo_ci = r.get('lo_ci', (0, 0))
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
            'Eff_Obs': r.get('n_obs', 0),
            'Lo_CI_Low': lo_ci[0],
            'Lo_CI_High': lo_ci[1],
            'Bootstrap_CI_Low': bs[0],
            'Bootstrap_CI_High': bs[1],
            'MC_pvalue': mc,
            'WF_Mean_Sharpe': wf.get('mean_sharpe', 0),
            'WF_Pct_Positive': wf.get('pct_positive', 0),
            'WF_N_Folds': wf.get('n_folds', 0),
        })

    df = pd.DataFrame(summary_rows)
    out_path = os.path.join(RESULTS_DIR, 'success_pairs_full.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {out_path}")

    generate_success_pairs_figure(results)

    return df


def main():
    print("\n" + "=" * 70)
    print("  DAY 5: FULL-SCALE SUCCESS PAIR VALIDATION")
    print("=" * 70)

    summary_path = os.path.join(RESULTS_DIR, 'expanded_universe_summary.csv')
    pairs = select_success_pairs(summary_path)

    print("\n  Selected success pairs:")
    for ty, tx in pairs:
        print(f"    - {ty}/{tx}")

    run_full_pipeline(pairs, quick=False)


if __name__ == '__main__':
    main()
