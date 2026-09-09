import numpy as np
import pandas as pd
import os
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, ConservativeKalman, StrictRegimeClassifier,
    calculate_zscore, generate_signals, download_data
)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. REGIME TIMING ALPHA
# =============================================================================

def compute_regime_timing_alpha(
    stock_y: pd.Series, stock_x: pd.Series, market_index: pd.Series,
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
) -> dict:
    """
    Regime timing alpha = Return(with regime gating) − Return(no regime gating).

    The regime classifier blocks entries during CRISIS and tightens parameters
    during VOLATILE periods.  This measures the value of that gating.
    """
    # Full system (with regime gating)
    sys_full = ConservativeSystem()
    res_full = sys_full.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=2.0, exit_z_volatile=0.3,
        min_hold_days=3, max_hold_days=30, stop_loss_mult=2.0,
        z_score_window=40, slippage_bps=slippage_bps, verbose=False,
    )
    oos_full = res_full[res_full['period'] == 'oos']
    ret_full = oos_full['strategy_return'].dropna()

    # No regime gating (always NORMAL)
    sys_no_regime = ConservativeSystem()
    res_no_regime = sys_no_regime.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=1.5, exit_z_volatile=0.5,  # same as normal
        min_hold_days=3, max_hold_days=30, stop_loss_mult=2.0,
        z_score_window=40, slippage_bps=slippage_bps, verbose=False,
    )

    # Force all regimes to NORMAL in the no-regime run by using baseline_return
    oos_no_regime = res_no_regime[res_no_regime['period'] == 'oos']
    ret_no_regime = oos_no_regime['baseline_return'].dropna()

    total_full = (1 + ret_full).prod() - 1
    total_no_regime = (1 + ret_no_regime).prod() - 1

    # Sharpe
    def sharpe(r):
        if len(r) == 0 or r.std() == 0:
            return 0.0
        tr = (1 + r).prod() - 1
        ar = (1 + tr) ** (252 / len(r)) - 1
        av = r.std() * np.sqrt(252)
        return ar / av if av > 0 else 0.0

    sharpe_full = sharpe(ret_full)
    sharpe_no_regime = sharpe(ret_no_regime)

    # Break down by regime
    regime_returns = {}
    for r_val, r_name in {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}.items():
        mask = oos_full['regime'] == r_val
        if mask.sum() > 0:
            r_days = oos_full.loc[mask, 'strategy_return'].dropna()
            regime_returns[r_name] = {
                'days': int(mask.sum()),
                'total_return_%': round((1 + r_days).prod() - 1, 4) * 100 if len(r_days) > 0 else 0,
                'avg_daily_%': round(r_days.mean() * 100, 4) if len(r_days) > 0 else 0,
            }

    return {
        'total_return_with_regime_%': round(total_full * 100, 2),
        'total_return_no_regime_%': round(total_no_regime * 100, 2),
        'regime_timing_alpha_%': round((total_full - total_no_regime) * 100, 2),
        'sharpe_with_regime': round(sharpe_full, 4),
        'sharpe_no_regime': round(sharpe_no_regime, 4),
        'sharpe_alpha': round(sharpe_full - sharpe_no_regime, 4),
        'regime_breakdown': regime_returns,
    }


# =============================================================================
# 2. MEAN REVERSION ALPHA
# =============================================================================

def compute_mean_reversion_alpha(
    stock_y: pd.Series, stock_x: pd.Series, market_index: pd.Series,
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    n_random: int = 1000,
    random_seed: int = 42,
) -> dict:
    """
    Mean reversion alpha = Return(z-score entries) − Return(random entries).

    Random entries are constrained to the SAME tradeable days (NORMAL/VOLATILE)
    so the comparison isolates signal quality, not regime timing.
    """
    rng = np.random.RandomState(random_seed)

    # Full system run
    sys_full = ConservativeSystem()
    res_full = sys_full.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        slippage_bps=slippage_bps, verbose=False,
    )
    oos = res_full[res_full['period'] == 'oos'].copy()
    ret_full = oos['strategy_return'].dropna()
    total_full = (1 + ret_full).prod() - 1

    # Actual trade stats
    trades = sys_full.get_trade_analysis(period='oos')
    n_trades = len(trades)
    if n_trades < 2:
        return {
            'total_return_strategy_%': round(total_full * 100, 2),
            'avg_random_return_%': 0.0,
            'mean_reversion_alpha_%': round(total_full * 100, 2),
            'p_value': 1.0,
            'n_random': 0,
        }

    durations = trades['duration_days'].values
    mean_dur = max(int(np.mean(durations)), 2)
    std_dur = max(int(np.std(durations)), 1)

    spread_ret = oos['spread_return'].dropna().values
    tradeable_mask = (oos['regime'].values >= 1)  # NORMAL or VOLATILE, not CRISIS
    n_days = len(spread_ret)

    random_returns = np.zeros(n_random)

    for s in range(n_random):
        sim_signal = np.zeros(n_days)
        i = 0
        placed = 0

        while placed < n_trades and i < n_days - 2:
            gap = rng.geometric(p=max(n_trades / n_days, 0.01))
            i += gap
            if i >= n_days - 2:
                break

            # Only enter on tradeable days
            found = False
            for offset in range(min(20, n_days - i)):
                if i + offset < n_days and tradeable_mask[i + offset]:
                    i = i + offset
                    found = True
                    break
            if not found:
                i += 5
                continue

            hold = max(2, int(rng.normal(mean_dur, std_dur)))
            hold = min(hold, n_days - i - 1)
            direction = rng.choice([-1, 1])
            sim_signal[i:i + hold] = direction
            i += hold
            placed += 1

        sim_daily = np.roll(sim_signal, 1) * spread_ret
        sim_daily[0] = 0
        cost = np.abs(np.diff(np.concatenate([[0], sim_signal]))) * (slippage_bps / 10_000)
        sim_daily -= cost
        random_returns[s] = np.prod(1 + sim_daily) - 1

    avg_random = random_returns.mean()
    p_value = (random_returns >= total_full).mean()

    return {
        'total_return_strategy_%': round(total_full * 100, 2),
        'avg_random_return_%': round(avg_random * 100, 2),
        'mean_reversion_alpha_%': round((total_full - avg_random) * 100, 2),
        'p_value': round(p_value, 4),
        'random_return_std_%': round(random_returns.std() * 100, 2),
        'n_random': n_random,
        'random_sharpe_mean': round(np.mean([
            ((1 + r) ** (252 / n_days) - 1) / max(np.std(np.random.choice(spread_ret, n_days)) * np.sqrt(252), 1e-8)
            for r in random_returns[:100]
        ]), 4),
    }


# =============================================================================
# 3. POSITION SIZING ALPHA
# =============================================================================

def compute_position_sizing_alpha(
    stock_y: pd.Series, stock_x: pd.Series, market_index: pd.Series,
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
) -> dict:
    """
    Position sizing alpha = Return(actual signals with magnitude)
                          − Return(sign-only signals, fixed ±1).

    Our generate_signals already outputs ±1 signals, so the main sizing
    effect comes from the transaction cost model where signal CHANGE magnitude
    drives slippage.  We compare the actual system vs a "always max position" variant
    that snaps to full position immediately (no gradual entry).
    """
    # Full system
    sys_full = ConservativeSystem()
    res_full = sys_full.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        slippage_bps=slippage_bps, verbose=False,
    )
    oos_full = res_full[res_full['period'] == 'oos'].copy()
    ret_full = oos_full['strategy_return'].dropna()
    total_full = (1 + ret_full).prod() - 1

    # Compare: use same signals but different holding period (no min hold)
    sys_no_hold = ConservativeSystem()
    res_no_hold = sys_no_hold.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        min_hold_days=0,
        slippage_bps=slippage_bps, verbose=False,
    )
    oos_no_hold = res_no_hold[res_no_hold['period'] == 'oos'].copy()
    ret_no_hold = oos_no_hold['strategy_return'].dropna()
    total_no_hold = (1 + ret_no_hold).prod() - 1

    # Compare: aggressive sizing (lower entry threshold → more active)
    sys_aggressive = ConservativeSystem()
    res_aggressive = sys_aggressive.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        entry_z_normal=1.0, exit_z_normal=0.3,
        entry_z_volatile=1.5, exit_z_volatile=0.2,
        min_hold_days=2, max_hold_days=30,
        slippage_bps=slippage_bps, verbose=False,
    )
    oos_aggressive = res_aggressive[res_aggressive['period'] == 'oos'].copy()
    ret_aggressive = oos_aggressive['strategy_return'].dropna()
    total_aggressive = (1 + ret_aggressive).prod() - 1

    def sharpe(r):
        if len(r) == 0 or r.std() == 0:
            return 0.0
        tr = (1 + r).prod() - 1
        ar = (1 + tr) ** (252 / len(r)) - 1
        av = r.std() * np.sqrt(252)
        return ar / av if av > 0 else 0.0

    n_trades_full = (oos_full['final_signal'].diff().abs() > 0).sum() // 2
    n_trades_no_hold = (oos_no_hold['final_signal'].diff().abs() > 0).sum() // 2
    n_trades_aggressive = (oos_aggressive['final_signal'].diff().abs() > 0).sum() // 2

    return {
        'conservative_return_%': round(total_full * 100, 2),
        'no_hold_return_%': round(total_no_hold * 100, 2),
        'aggressive_return_%': round(total_aggressive * 100, 2),
        'conservative_sharpe': round(sharpe(ret_full), 4),
        'no_hold_sharpe': round(sharpe(ret_no_hold), 4),
        'aggressive_sharpe': round(sharpe(ret_aggressive), 4),
        'sizing_alpha_%': round((total_full - total_aggressive) * 100, 2),
        'hold_alpha_%': round((total_full - total_no_hold) * 100, 2),
        'trades_conservative': int(n_trades_full),
        'trades_no_hold': int(n_trades_no_hold),
        'trades_aggressive': int(n_trades_aggressive),
    }


# =============================================================================
# 4. KALMAN FILTER ALPHA
# =============================================================================

def compute_kalman_alpha(
    stock_y: pd.Series, stock_x: pd.Series, market_index: pd.Series,
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
) -> dict:
    """
    Kalman alpha = Return(adaptive Kalman hedge) − Return(fixed OLS hedge).

    The adaptive Kalman hedge ratio continuously updates, responding to regime
    shifts and structural breaks.  Fixed OLS uses the training-period estimate.
    """
    from sklearn.linear_model import LinearRegression

    # Full system (Kalman)
    sys_kalman = ConservativeSystem()
    res_kalman = sys_kalman.run_backtest(
        stock_y, stock_x, market_index,
        train_end_date=train_end_date,
        slippage_bps=slippage_bps, verbose=False,
    )
    oos_kalman = res_kalman[res_kalman['period'] == 'oos']
    ret_kalman = oos_kalman['strategy_return'].dropna()

    # OLS baseline (from baselines.py approach)
    data = pd.DataFrame({'Y': stock_y, 'X': stock_x, 'Market': market_index}).dropna()
    train_end = pd.Timestamp(train_end_date)
    train = data[data.index <= train_end]

    model = LinearRegression()
    model.fit(train['X'].values.reshape(-1, 1), train['Y'].values)
    ols_hedge = model.coef_[0]
    ols_intercept = model.intercept_

    spread_ols = data['Y'] - ols_hedge * data['X'] - ols_intercept
    z_ols = calculate_zscore(spread_ols, 40)

    # Use regime from Kalman run for fair comparison
    data['regime'] = res_kalman['regime'].reindex(data.index).fillna(2).astype(int)

    sig_ols = generate_signals(
        z_ols, data['regime'],
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=2.0, exit_z_volatile=0.3,
        min_hold_days=3, max_hold_days=30, stop_loss_mult=2.0,
    )

    data['returns_Y'] = data['Y'].pct_change()
    data['returns_X'] = data['X'].pct_change()
    data['spread_return_ols'] = data['returns_Y'] - ols_hedge * data['returns_X']
    data['ols_return'] = sig_ols.shift(1) * data['spread_return_ols']
    cost_ols = sig_ols.diff().abs().fillna(0) * (slippage_bps / 10_000)
    data['ols_return_net'] = data['ols_return'] - cost_ols

    oos_mask = data.index > train_end
    ret_ols = data.loc[oos_mask, 'ols_return_net'].dropna()

    def sharpe(r):
        if len(r) == 0 or r.std() == 0:
            return 0.0
        tr = (1 + r).prod() - 1
        ar = (1 + tr) ** (252 / len(r)) - 1
        av = r.std() * np.sqrt(252)
        return ar / av if av > 0 else 0.0

    total_kalman = (1 + ret_kalman).prod() - 1
    total_ols = (1 + ret_ols).prod() - 1

    # Hedge ratio statistics
    oos_hedges = res_kalman.loc[res_kalman['period'] == 'oos', 'hedge_ratio']

    return {
        'kalman_return_%': round(total_kalman * 100, 2),
        'ols_return_%': round(total_ols * 100, 2),
        'kalman_alpha_%': round((total_kalman - total_ols) * 100, 2),
        'kalman_sharpe': round(sharpe(ret_kalman), 4),
        'ols_sharpe': round(sharpe(ret_ols), 4),
        'sharpe_alpha': round(sharpe(ret_kalman) - sharpe(ret_ols), 4),
        'hedge_ratio_mean': round(oos_hedges.mean(), 4),
        'hedge_ratio_std': round(oos_hedges.std(), 4),
        'ols_hedge_ratio': round(ols_hedge, 4),
    }


# =============================================================================
# 5. FULL ATTRIBUTION DECOMPOSITION
# =============================================================================

def run_full_attribution(
    ticker_y: str, ticker_x: str,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    n_random: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Run complete performance attribution for a single pair.

    Returns dict with 4 alpha components and a summary decomposition.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  PERFORMANCE ATTRIBUTION: {ticker_y}/{ticker_x}")
        print(f"  Train ≤ {train_end_date}")
        print(f"{'=' * 70}")

    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    attribution = {}

    # 1. Regime Timing Alpha
    if verbose:
        print("\n  [1/4] Regime Timing Alpha...")
    attribution['regime_timing'] = compute_regime_timing_alpha(
        stock_y, stock_x, market, train_end_date, slippage_bps
    )
    if verbose:
        rt = attribution['regime_timing']
        print(f"    With regime: {rt['total_return_with_regime_%']:+.2f}%")
        print(f"    No regime:   {rt['total_return_no_regime_%']:+.2f}%")
        print(f"    Alpha:       {rt['regime_timing_alpha_%']:+.2f}%")

    # 2. Mean Reversion Alpha
    if verbose:
        print("\n  [2/4] Mean Reversion Alpha...")
    attribution['mean_reversion'] = compute_mean_reversion_alpha(
        stock_y, stock_x, market, train_end_date, slippage_bps,
        n_random=n_random,
    )
    if verbose:
        mr = attribution['mean_reversion']
        print(f"    Strategy:    {mr['total_return_strategy_%']:+.2f}%")
        print(f"    Avg random:  {mr['avg_random_return_%']:+.2f}%")
        print(f"    Alpha:       {mr['mean_reversion_alpha_%']:+.2f}%")
        print(f"    p-value:     {mr['p_value']:.4f}")

    # 3. Position Sizing Alpha
    if verbose:
        print("\n  [3/4] Position Sizing / Trade Management Alpha...")
    attribution['position_sizing'] = compute_position_sizing_alpha(
        stock_y, stock_x, market, train_end_date, slippage_bps
    )
    if verbose:
        ps = attribution['position_sizing']
        print(f"    Conservative: {ps['conservative_return_%']:+.2f}% "
              f"(Sharpe={ps['conservative_sharpe']:.3f}, {ps['trades_conservative']} trades)")
        print(f"    No hold:      {ps['no_hold_return_%']:+.2f}% "
              f"(Sharpe={ps['no_hold_sharpe']:.3f}, {ps['trades_no_hold']} trades)")
        print(f"    Aggressive:   {ps['aggressive_return_%']:+.2f}% "
              f"(Sharpe={ps['aggressive_sharpe']:.3f}, {ps['trades_aggressive']} trades)")

    # 4. Kalman Filter Alpha
    if verbose:
        print("\n  [4/4] Adaptive Kalman Alpha...")
    attribution['kalman'] = compute_kalman_alpha(
        stock_y, stock_x, market, train_end_date, slippage_bps
    )
    if verbose:
        ka = attribution['kalman']
        print(f"    Kalman:  {ka['kalman_return_%']:+.2f}% (Sharpe={ka['kalman_sharpe']:.3f})")
        print(f"    OLS:     {ka['ols_return_%']:+.2f}% (Sharpe={ka['ols_sharpe']:.3f})")
        print(f"    Alpha:   {ka['kalman_alpha_%']:+.2f}%")
        print(f"    Hedge σ: {ka['hedge_ratio_std']:.4f} (OLS fixed: {ka['ols_hedge_ratio']:.4f})")

    # Summary decomposition
    summary = {
        'Regime Timing': attribution['regime_timing']['regime_timing_alpha_%'],
        'Mean Reversion Signal': attribution['mean_reversion']['mean_reversion_alpha_%'],
        'Position Sizing': attribution['position_sizing']['sizing_alpha_%'],
        'Adaptive Kalman': attribution['kalman']['kalman_alpha_%'],
    }

    attribution['summary'] = summary

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  ATTRIBUTION SUMMARY ({ticker_y}/{ticker_x})")
        print(f"{'=' * 70}")
        for source, alpha in summary.items():
            bar = "+" * max(0, int(alpha)) + "-" * max(0, int(-alpha))
            print(f"    {source:.<30} {alpha:+.2f}% {bar}")
        total_alpha = sum(summary.values())
        print(f"    {'Sum of Alphas':.<30} {total_alpha:+.2f}%")

    # Save chart
    try:
        _save_attribution_chart(summary, ticker_y, ticker_x)
        if verbose:
            print(f"\n  📊 Chart saved: Research/results/attribution_{ticker_y}_{ticker_x}.png")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Chart save failed: {e}")

    return attribution


def _save_attribution_chart(summary: dict, ticker_y: str, ticker_x: str):
    """Save stacked bar chart of return attribution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = list(summary.keys())
    values = list(summary.values())
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked horizontal bar
    pos_cum = 0
    neg_cum = 0
    for src, val, col in zip(sources, values, colors):
        if val >= 0:
            ax.barh(0, val, left=pos_cum, height=0.5, color=col,
                    label=f"{src}: {val:+.2f}%", edgecolor='white', linewidth=1)
            ax.text(pos_cum + val / 2, 0, f"{val:+.1f}%",
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            pos_cum += val
        else:
            ax.barh(0, val, left=neg_cum, height=0.5, color=col,
                    label=f"{src}: {val:+.2f}%", edgecolor='white', linewidth=1)
            ax.text(neg_cum + val / 2, 0, f"{val:+.1f}%",
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            neg_cum += val

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks([])
    ax.set_xlabel('Alpha Contribution (%)', fontsize=12)
    ax.set_title(f'Performance Attribution — {ticker_y}/{ticker_x} (OOS)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'attribution_{ticker_y}_{ticker_x}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MULTI-PAIR ATTRIBUTION
# =============================================================================

def run_attribution_multi_pair(
    pairs: list,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run attribution for multiple pairs and aggregate.

    Parameters
    ----------
    pairs : list of (ticker_y, ticker_x) tuples

    Returns
    -------
    pd.DataFrame : rows = pairs, columns = alpha sources
    """
    rows = []

    for ty, tx in pairs:
        try:
            result = run_full_attribution(
                ty, tx, start_date, end_date, train_end_date,
                slippage_bps, verbose=verbose,
            )
            row = {'Pair': f"{ty}/{tx}"}
            row.update(result['summary'])
            rows.append(row)
        except Exception as e:
            logger.warning(f"Attribution failed for {ty}/{tx}: {e}")

    df = pd.DataFrame(rows)
    if len(df) > 0 and verbose:
        print(f"\n{'=' * 70}")
        print(f"  MULTI-PAIR ATTRIBUTION SUMMARY")
        print(f"{'=' * 70}")
        for col in ['Regime Timing', 'Mean Reversion Signal',
                     'Position Sizing', 'Adaptive Kalman']:
            if col in df.columns:
                avg = df[col].mean()
                print(f"  {col:.<35} avg={avg:+.2f}%")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    result = run_full_attribution('BAC', 'PNC', verbose=True)
