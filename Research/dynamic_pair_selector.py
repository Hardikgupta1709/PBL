"""
Dynamic Pair Selection Filter
=================================
Week 8 Improvement: Real-time cointegration health gating.

Pairs are only traded when they pass live rolling health checks:
    1. ADF p-value < threshold (spread is stationary)
    2. Hurst exponent < 0.5 (pair is mean-reverting)
    3. Cointegration p-value < threshold (relationship intact)

The filter uses ONLY trailing data — no look-ahead bias.
It integrates as a post-processing layer on top of ConservativeSystem
signals, forcing positions to zero when pair health degrades.

Design:
    - PairHealthMonitor: computes rolling health for a single pair
    - DynamicPairSelector: manages health across a universe of pairs
    - backtest_with_dynamic_filter(): wraps ConservativeSystem with health gating
    - compare_filtered_vs_unfiltered(): ablation comparison

References:
    Gatev et al. (2006) — pairs selection with distance metric
    Krauss (2017) — "Statistical Arbitrage Pairs Trading Strategies"
    Do & Faff (2010) — cointegration breakdown and pair selection
"""

import numpy as np
import pandas as pd
import os
import sys
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, ConservativeKalman, download_data,
    calculate_zscore, calculate_half_life, generate_signals
)
from Research.cointegration_analysis import (
    hurst_exponent, rolling_cointegration, rolling_hurst,
    rolling_adf, rolling_half_life, rolling_hedge_ratio
)
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# =============================================================================
# 1. PAIR HEALTH MONITOR
# =============================================================================

@dataclass
class HealthThresholds:
    """Configurable thresholds for pair health gating."""
    adf_pvalue: float = 0.05        # Spread stationarity threshold
    hurst_max: float = 0.50         # Maximum Hurst for mean-reversion
    coint_pvalue: float = 0.05      # Cointegration threshold
    half_life_max: float = 60.0     # Maximum half-life (days)
    half_life_min: float = 3.0      # Minimum half-life (too fast = noise)
    hedge_ratio_max_drift: float = 0.3  # Max hedge ratio std over window
    lookback_window: int = 126      # Rolling window (6 months)
    min_checks_passed: int = 2      # Min criteria to be "healthy" (out of 3 core)
    grace_period: int = 5           # Days to wait before re-entering after unhealthy


class PairHealthMonitor:
    """
    Real-time pair health monitoring using rolling statistical tests.

    Computes at each time step (using only past data):
        - Rolling ADF p-value on spread
        - Rolling Hurst exponent on spread
        - Rolling cointegration p-value between Y and X
        - Rolling half-life of mean reversion
        - Rolling hedge ratio stability

    Returns a boolean health signal: True = OK to trade, False = pair broken.
    """

    def __init__(self, thresholds: Optional[HealthThresholds] = None):
        self.thresholds = thresholds or HealthThresholds()

    def compute_rolling_health(
        self,
        stock_y: pd.Series,
        stock_x: pd.Series,
        spread: pd.Series,
        step: int = 1,
    ) -> pd.DataFrame:
        """
        Compute rolling pair health metrics at every time step.

        Uses ONLY trailing data (no look-ahead). At time t, the health
        check uses data from [t - window, t].

        Parameters
        ----------
        stock_y, stock_x : pd.Series
            Price series for the pair.
        spread : pd.Series
            Spread series (from Kalman filter or OLS).
        step : int
            Compute health every `step` days (1 = daily, 5 = weekly).

        Returns
        -------
        DataFrame with columns:
            adf_pvalue, hurst, coint_pvalue, half_life,
            hedge_ratio_std, is_healthy, checks_passed, health_score
        """
        window = self.thresholds.lookback_window
        n = len(spread)
        data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()

        # Pre-align all series to common index
        common_idx = spread.index.intersection(data.index)
        spread = spread.reindex(common_idx)
        data = data.reindex(common_idx)
        n = len(common_idx)

        rows = []
        for i in range(0, n, step):
            date = common_idx[i]

            if i < window:
                # Not enough data yet — assume healthy (default)
                rows.append({
                    'date': date,
                    'adf_pvalue': np.nan,
                    'hurst': np.nan,
                    'coint_pvalue': np.nan,
                    'half_life': np.nan,
                    'hedge_ratio_std': np.nan,
                    'checks_passed': 3,  # Assume healthy before enough data
                    'health_score': 1.0,
                    'is_healthy': True,
                })
                continue

            window_spread = spread.iloc[i - window:i]
            window_y = data['Y'].iloc[i - window:i]
            window_x = data['X'].iloc[i - window:i]

            checks_passed = 0
            metrics = {}

            # --- Check 1: ADF stationarity of spread ---
            try:
                adf_stat, adf_p, *_ = adfuller(
                    window_spread.dropna().values, maxlag=15
                )
                metrics['adf_pvalue'] = adf_p
                if adf_p < self.thresholds.adf_pvalue:
                    checks_passed += 1
            except Exception:
                metrics['adf_pvalue'] = np.nan

            # --- Check 2: Hurst exponent ---
            try:
                H = hurst_exponent(window_spread.dropna().values)
                metrics['hurst'] = H
                if H < self.thresholds.hurst_max:
                    checks_passed += 1
            except Exception:
                metrics['hurst'] = np.nan

            # --- Check 3: Cointegration test ---
            try:
                _, coint_p, _ = coint(
                    window_y.dropna().values,
                    window_x.dropna().values
                )
                metrics['coint_pvalue'] = coint_p
                if coint_p < self.thresholds.coint_pvalue:
                    checks_passed += 1
            except Exception:
                metrics['coint_pvalue'] = np.nan

            # --- Supplementary: Half-life ---
            try:
                hl = calculate_half_life(window_spread)
                metrics['half_life'] = hl
            except Exception:
                metrics['half_life'] = np.nan

            # --- Supplementary: Hedge ratio stability ---
            try:
                # Compute hedge ratios over sub-windows within the main window
                sub_window = max(30, window // 4)
                hedge_ratios = []
                for j in range(sub_window, len(window_y), sub_window // 2):
                    sub_y = window_y.iloc[j - sub_window:j]
                    sub_x = window_x.iloc[j - sub_window:j]
                    model = LinearRegression()
                    model.fit(sub_x.values.reshape(-1, 1), sub_y.values)
                    hedge_ratios.append(model.coef_[0])

                if len(hedge_ratios) >= 2:
                    metrics['hedge_ratio_std'] = np.std(hedge_ratios)
                else:
                    metrics['hedge_ratio_std'] = np.nan
            except Exception:
                metrics['hedge_ratio_std'] = np.nan

            # --- Health score (continuous, 0-1) ---
            # Combine all metrics into a single score
            score_components = []
            if not np.isnan(metrics.get('adf_pvalue', np.nan)):
                # Lower ADF p → better → score = 1 - p
                score_components.append(
                    max(0, 1.0 - metrics['adf_pvalue'] / 0.10)
                )
            if not np.isnan(metrics.get('hurst', np.nan)):
                # Lower Hurst → better → score = 1 - 2*H (clamped 0-1)
                score_components.append(
                    max(0, min(1, 1.0 - 2.0 * metrics['hurst']))
                )
            if not np.isnan(metrics.get('coint_pvalue', np.nan)):
                score_components.append(
                    max(0, 1.0 - metrics['coint_pvalue'] / 0.10)
                )

            health_score = (
                np.mean(score_components) if score_components else 0.5
            )

            is_healthy = checks_passed >= self.thresholds.min_checks_passed

            rows.append({
                'date': date,
                'adf_pvalue': metrics.get('adf_pvalue', np.nan),
                'hurst': metrics.get('hurst', np.nan),
                'coint_pvalue': metrics.get('coint_pvalue', np.nan),
                'half_life': metrics.get('half_life', np.nan),
                'hedge_ratio_std': metrics.get('hedge_ratio_std', np.nan),
                'checks_passed': checks_passed,
                'health_score': health_score,
                'is_healthy': is_healthy,
            })

        health_df = pd.DataFrame(rows).set_index('date')

        # Forward-fill for steps > 1 so every trading day has a health value
        if step > 1:
            health_df = health_df.reindex(common_idx, method='ffill')

        return health_df

    def compute_efficient_health(
        self,
        stock_y: pd.Series,
        stock_x: pd.Series,
        spread: pd.Series,
        step: int = 5,
    ) -> pd.DataFrame:
        """
        Faster version: compute health every `step` days and forward-fill.
        Suitable for large-scale universe backtests.
        """
        return self.compute_rolling_health(stock_y, stock_x, spread, step=step)


# =============================================================================
# 2. SIGNAL GATING (APPLY HEALTH FILTER TO SIGNALS)
# =============================================================================

def apply_health_gate(
    signals: pd.Series,
    health: pd.DataFrame,
    grace_period: int = 5,
) -> pd.Series:
    """
    Apply pair health gating to trading signals.

    When the pair is unhealthy:
        - Block new entries
        - Force exit of existing positions after grace period

    Parameters
    ----------
    signals : pd.Series
        Raw trading signals from generate_signals() (values: -1, 0, 1).
    health : pd.DataFrame
        Output from PairHealthMonitor.compute_rolling_health().
        Must have 'is_healthy' column aligned to same index.
    grace_period : int
        Days to wait before forcing exit when pair becomes unhealthy.

    Returns
    -------
    pd.Series
        Filtered signals with health gating applied.
    """
    # Align health to signal index
    is_healthy = health['is_healthy'].reindex(signals.index, method='ffill').fillna(True)

    filtered = signals.copy()
    unhealthy_count = 0
    position = 0

    for i in range(len(filtered)):
        sig = signals.iloc[i]
        healthy = is_healthy.iloc[i]

        if not healthy:
            unhealthy_count += 1

            # Block new entries
            if position == 0:
                filtered.iloc[i] = 0
            # Force exit after grace period
            elif unhealthy_count > grace_period:
                filtered.iloc[i] = 0
                position = 0
            else:
                # Keep existing position during grace period
                filtered.iloc[i] = position
        else:
            unhealthy_count = 0
            position = sig

        if filtered.iloc[i] != 0:
            position = filtered.iloc[i]
        else:
            position = 0

    return filtered


def _health_score_to_size(
    score: float,
    low: float,
    mid: float,
    high: float,
    size_low: float,
    size_mid: float,
    size_high: float,
) -> float:
    """Map health score to position size (piecewise rule)."""
    if score < low:
        return 0.0
    if score < mid:
        return size_low
    if score < high:
        return size_mid
    return size_high


def apply_health_sizing_to_signals(
    signals: pd.Series,
    health: pd.DataFrame,
    sizing_thresholds: Tuple[float, float, float] = (0.5, 0.7, 0.85),
    sizing_values: Tuple[float, float, float] = (0.3, 0.6, 1.0),
) -> pd.Series:
    """
    Scale signals by health score using a piecewise sizing rule.

    This preserves direction but adjusts position size based on
    the current health score.
    """
    health_score = health['health_score'].reindex(
        signals.index, method='ffill'
    ).fillna(1.0)

    low, mid, high = sizing_thresholds
    size_low, size_mid, size_high = sizing_values
    sizes = health_score.apply(
        _health_score_to_size,
        args=(low, mid, high, size_low, size_mid, size_high),
    )
    return signals * sizes


# =============================================================================
# 3. BACKTEST WITH DYNAMIC FILTER (FULL PIPELINE)
# =============================================================================

def backtest_with_dynamic_filter(
    stock_y: pd.Series,
    stock_x: pd.Series,
    market: pd.Series,
    train_end_date: str = '2020-12-31',
    thresholds: Optional[HealthThresholds] = None,
    health_step: int = 5,
    apply_health_sizing: bool = False,
    sizing_thresholds: Tuple[float, float, float] = (0.5, 0.7, 0.85),
    sizing_values: Tuple[float, float, float] = (0.3, 0.6, 1.0),
    verbose: bool = True,
    **backtest_kwargs,
) -> Dict:
    """
    Run ConservativeSystem backtest with dynamic pair health gating.

    Steps:
        1. Run standard backtest to get spread, signals, returns
        2. Compute rolling pair health using PairHealthMonitor
        3. Apply health gate to filter out signals during unhealthy periods
        4. Recompute returns with filtered signals

    Parameters
    ----------
    stock_y, stock_x : pd.Series
        Price series.
    market : pd.Series
        Market benchmark (SPY).
    train_end_date : str
        End of training period.
    thresholds : HealthThresholds
        Health gating thresholds.
    health_step : int
        Compute health metrics every N days (5 = weekly).
    verbose : bool
        Print progress.
    **backtest_kwargs
        Additional args passed to ConservativeSystem.run_backtest().

    Returns
    -------
    dict with keys:
        - 'unfiltered': raw backtest results (pd.DataFrame)
        - 'filtered': filtered backtest results (pd.DataFrame)
        - 'health': health metrics (pd.DataFrame)
        - 'system': ConservativeSystem instance
        - 'metrics_unfiltered': performance dict
        - 'metrics_filtered': performance dict
        - 'filter_stats': statistics about the filter's impact
    """
    if thresholds is None:
        thresholds = HealthThresholds()

    # --- Step 1: Run standard backtest ---
    if verbose:
        print(f"\n  Step 1: Running standard backtest...")

    system = ConservativeSystem()
    bt_data = system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        verbose=verbose,
        **backtest_kwargs,
    )

    # --- Step 2: Compute pair health ---
    if verbose:
        print(f"\n  Step 2: Computing rolling pair health (window={thresholds.lookback_window}d, step={health_step}d)...")

    monitor = PairHealthMonitor(thresholds)
    health = monitor.compute_rolling_health(
        stock_y, stock_x, bt_data['spread'],
        step=health_step,
    )

    # --- Step 3: Apply health gate to signals ---
    if verbose:
        print(f"\n  Step 3: Applying health gate to signals...")

    raw_signals = bt_data['final_signal']
    filtered_signals = apply_health_gate(
        raw_signals, health,
        grace_period=thresholds.grace_period,
    )

    if apply_health_sizing:
        filtered_signals = apply_health_sizing_to_signals(
            filtered_signals,
            health,
            sizing_thresholds=sizing_thresholds,
            sizing_values=sizing_values,
        )

    # --- Step 4: Recompute returns with filtered signals ---
    if verbose:
        print(f"\n  Step 4: Recomputing returns with filtered signals...")

    filtered_bt = bt_data.copy()
    filtered_bt['final_signal_unfiltered'] = raw_signals
    filtered_bt['final_signal'] = filtered_signals

    # Recompute strategy returns
    filtered_bt['strategy_return_gross'] = (
        filtered_signals.shift(1) * bt_data['spread_return']
    )

    # Recompute transaction costs
    slippage_bps = backtest_kwargs.get('slippage_bps', 5.0)
    commission_per_trade = backtest_kwargs.get('commission_per_trade', 1.0)
    notional_per_leg = backtest_kwargs.get('notional_per_leg', 50000.0)

    signal_changes = filtered_signals.diff().abs().fillna(0)
    slippage_cost = signal_changes * (slippage_bps / 10_000)
    trades_occurred = (signal_changes > 0).astype(float)
    commission_cost = trades_occurred * (commission_per_trade * 2) / notional_per_leg

    # Market impact (simplified)
    daily_vol = bt_data['returns_Y'].rolling(20).std().fillna(
        bt_data['returns_Y'].std()
    )
    impact_per_trade = 0.1 * daily_vol * np.sqrt(notional_per_leg / 200_000_000)
    market_impact_cost = trades_occurred * impact_per_trade

    filtered_bt['cost_slippage'] = slippage_cost
    filtered_bt['cost_commission'] = commission_cost
    filtered_bt['cost_impact'] = market_impact_cost
    filtered_bt['transaction_cost'] = slippage_cost + commission_cost + market_impact_cost

    filtered_bt['strategy_return'] = (
        filtered_bt['strategy_return_gross'] - filtered_bt['transaction_cost']
    )
    filtered_bt['strategy_cumulative'] = (
        1 + filtered_bt['strategy_return']
    ).cumprod()

    # --- Step 5: Compute metrics ---
    train_end = pd.Timestamp(train_end_date)

    def compute_period_metrics(data, period_name):
        """Compute metrics for a given period."""
        if period_name == 'oos':
            subset = data[data['period'] == 'oos']
        elif period_name == 'train':
            subset = data[data['period'] == 'train']
        else:
            subset = data

        ret = subset['strategy_return'].dropna()
        if len(ret) == 0 or ret.std() == 0:
            return {
                'total_return': 0, 'annual_return': 0, 'annual_vol': 0,
                'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'n_days': 0,
                'n_trades': 0,
            }

        total_ret = (1 + ret).prod() - 1
        n = len(ret)
        ann_ret = (1 + total_ret) ** (252 / n) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + ret).cumprod()
        max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        win_rate = (ret > 0).mean()

        # Trade count
        sig_changes = subset['final_signal'].diff().abs()
        n_trades = int((sig_changes > 0).sum()) // 2

        return {
            'total_return': total_ret,
            'annual_return': ann_ret,
            'annual_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'n_days': n,
            'n_trades': n_trades,
        }

    metrics_unfiltered = {
        'oos': compute_period_metrics(bt_data, 'oos'),
        'train': compute_period_metrics(bt_data, 'train'),
    }
    metrics_filtered = {
        'oos': compute_period_metrics(filtered_bt, 'oos'),
        'train': compute_period_metrics(filtered_bt, 'train'),
    }

    # --- Step 6: Filter statistics ---
    oos_mask = bt_data['period'] == 'oos'
    oos_health = health.reindex(bt_data.index[oos_mask], method='ffill')

    if len(oos_health) > 0 and 'is_healthy' in oos_health.columns:
        pct_healthy = oos_health['is_healthy'].mean() * 100
        avg_health_score = oos_health['health_score'].mean()
    else:
        pct_healthy = 100.0
        avg_health_score = 1.0

    # Compare signal activity
    raw_active = (raw_signals[oos_mask] != 0).sum()
    filtered_active = (filtered_signals[oos_mask] != 0).sum()
    signals_blocked_pct = (
        (raw_active - filtered_active) / max(raw_active, 1) * 100
    )

    filter_stats = {
        'oos_pct_healthy': pct_healthy,
        'oos_avg_health_score': avg_health_score,
        'oos_days_healthy': int(oos_health['is_healthy'].sum()) if len(oos_health) > 0 else 0,
        'oos_days_unhealthy': int((~oos_health['is_healthy']).sum()) if len(oos_health) > 0 else 0,
        'raw_active_days': int(raw_active),
        'filtered_active_days': int(filtered_active),
        'signals_blocked_pct': signals_blocked_pct,
        'sharpe_change': (
            metrics_filtered['oos']['sharpe'] - metrics_unfiltered['oos']['sharpe']
        ),
        'return_change': (
            metrics_filtered['oos']['total_return'] - metrics_unfiltered['oos']['total_return']
        ),
    }

    if apply_health_sizing:
        oos_abs = filtered_signals[oos_mask].abs()
        filter_stats['oos_avg_position_size'] = oos_abs.mean()

    # --- Print Summary ---
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  DYNAMIC PAIR FILTER — RESULTS")
        print(f"{'=' * 70}")

        print(f"\n  Pair Health (OOS):")
        print(f"    Healthy:     {pct_healthy:.1f}% of OOS days")
        print(f"    Avg Score:   {avg_health_score:.3f}")
        print(f"    Signals Blocked: {signals_blocked_pct:.1f}%")

        print(f"\n  {'Metric':<25} {'Unfiltered':>12} {'Filtered':>12} {'Change':>12}")
        print(f"  {'─' * 61}")

        for metric_name, key in [
            ('OOS Sharpe', 'sharpe'),
            ('OOS Return (%)', 'total_return'),
            ('OOS Max DD (%)', 'max_dd'),
            ('OOS Trades', 'n_trades'),
            ('OOS Win Rate', 'win_rate'),
        ]:
            uf = metrics_unfiltered['oos'][key]
            fi = metrics_filtered['oos'][key]

            if key in ('total_return', 'max_dd', 'win_rate'):
                print(f"  {metric_name:<25} {uf*100:>11.2f}% {fi*100:>11.2f}% "
                      f"{(fi-uf)*100:>+11.2f}%")
            elif key == 'n_trades':
                print(f"  {metric_name:<25} {uf:>12d} {fi:>12d} "
                      f"{fi-uf:>+12d}")
            else:
                print(f"  {metric_name:<25} {uf:>12.3f} {fi:>12.3f} "
                      f"{fi-uf:>+12.3f}")

        # Assessment
        sharpe_delta = filter_stats['sharpe_change']
        if sharpe_delta > 0.1:
            print(f"\n  ✅ Filter IMPROVED OOS Sharpe by {sharpe_delta:+.3f}")
        elif sharpe_delta > -0.05:
            print(f"\n  ⚠️ Filter had MINIMAL impact ({sharpe_delta:+.3f})")
        else:
            print(f"\n  ❌ Filter HURT performance ({sharpe_delta:+.3f})")
            print(f"     This pair may be trading during 'unhealthy' periods profitably")

    return {
        'unfiltered': bt_data,
        'filtered': filtered_bt,
        'health': health,
        'system': system,
        'metrics_unfiltered': metrics_unfiltered,
        'metrics_filtered': metrics_filtered,
        'filter_stats': filter_stats,
    }


# =============================================================================
# 4. DYNAMIC PAIR SELECTOR (UNIVERSE-LEVEL)
# =============================================================================

class DynamicPairSelector:
    """
    Universe-level dynamic pair selection.

    Given a pool of candidate pairs, this class:
        1. Computes rolling health for every pair
        2. Ranks pairs by health score at each time step
        3. Selects top-K healthiest pairs for trading
        4. Rotates pairs when health degrades

    This is used for Week 9 (expanded universe) where we trade
    a dynamic subset of the 20+ discovered pairs.
    """

    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        max_active_pairs: int = 5,
        min_health_to_enter: float = 0.6,
        min_health_to_keep: float = 0.4,
        rebalance_frequency: int = 5,  # days
    ):
        self.thresholds = thresholds or HealthThresholds()
        self.max_active_pairs = max_active_pairs
        self.min_health_to_enter = min_health_to_enter
        self.min_health_to_keep = min_health_to_keep
        self.rebalance_frequency = rebalance_frequency
        self.monitor = PairHealthMonitor(self.thresholds)

    def select_pairs(
        self,
        pair_health_dict: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
    ) -> List[str]:
        """
        Select which pairs to trade on a given date.

        Parameters
        ----------
        pair_health_dict : dict
            Maps pair_name → health DataFrame (output of compute_rolling_health).
        date : pd.Timestamp
            Current date.

        Returns
        -------
        list of pair names to trade.
        """
        scores = {}
        for pair_name, health_df in pair_health_dict.items():
            # Get health score on this date (or nearest before)
            valid = health_df[health_df.index <= date]
            if len(valid) == 0:
                continue
            latest = valid.iloc[-1]
            scores[pair_name] = {
                'health_score': latest['health_score'],
                'is_healthy': latest['is_healthy'],
                'checks_passed': latest['checks_passed'],
            }

        # Filter by minimum health
        eligible = {
            k: v for k, v in scores.items()
            if v['health_score'] >= self.min_health_to_enter and v['is_healthy']
        }

        # Rank by health score
        ranked = sorted(
            eligible.items(),
            key=lambda x: x[1]['health_score'],
            reverse=True,
        )

        # Select top K
        selected = [name for name, _ in ranked[:self.max_active_pairs]]
        return selected

    def compute_universe_health(
        self,
        pairs_data: Dict[str, Dict[str, pd.Series]],
        step: int = 5,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute health metrics for all pairs in the universe.

        Parameters
        ----------
        pairs_data : dict
            Maps pair_name → {'stock_y': Series, 'stock_x': Series, 'spread': Series}
        step : int
            Health computation step.
        verbose : bool
            Print progress.

        Returns
        -------
        dict mapping pair_name → health DataFrame
        """
        health_dict = {}
        n = len(pairs_data)

        for i, (pair_name, data) in enumerate(pairs_data.items()):
            if verbose:
                print(f"  [{i+1}/{n}] Computing health for {pair_name}...")

            try:
                health = self.monitor.compute_rolling_health(
                    data['stock_y'], data['stock_x'], data['spread'],
                    step=step,
                )
                health_dict[pair_name] = health
            except Exception as e:
                if verbose:
                    print(f"    ✗ Error: {e}")
                continue

        return health_dict

    def generate_rotation_schedule(
        self,
        pair_health_dict: Dict[str, pd.DataFrame],
        trading_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Generate a pair rotation schedule over the trading period.

        Returns DataFrame with columns: date, active_pairs (list), n_active, avg_health
        """
        rows = []
        rebal_counter = 0

        for date in trading_dates:
            rebal_counter += 1

            if rebal_counter >= self.rebalance_frequency or not rows:
                active = self.select_pairs(pair_health_dict, date)
                rebal_counter = 0

                # Compute average health of active pairs
                health_scores = []
                for pair in active:
                    if pair in pair_health_dict:
                        h = pair_health_dict[pair]
                        valid = h[h.index <= date]
                        if len(valid) > 0:
                            health_scores.append(valid.iloc[-1]['health_score'])

                avg_health = np.mean(health_scores) if health_scores else 0.0
            else:
                if rows:
                    active = rows[-1]['active_pairs']
                    avg_health = rows[-1]['avg_health']
                else:
                    active = []
                    avg_health = 0.0

            rows.append({
                'date': date,
                'active_pairs': active,
                'n_active': len(active),
                'avg_health': avg_health,
            })

        return pd.DataFrame(rows).set_index('date')


# =============================================================================
# 5. COMPARISON & ABLATION
# =============================================================================

def compare_filtered_vs_unfiltered(
    ticker_y: str,
    ticker_x: str,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    thresholds: Optional[HealthThresholds] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compare backtest results with and without dynamic pair health filter.

    Downloads data and runs the full comparison pipeline.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  DYNAMIC FILTER ABLATION: {ticker_y}/{ticker_x}")
        print(f"  {start_date} → {end_date} | Train ≤ {train_end_date}")
        print(f"{'=' * 70}")

    # Download data
    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    # Run filtered backtest
    result = backtest_with_dynamic_filter(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        thresholds=thresholds,
        verbose=verbose,
    )

    return result


# =============================================================================
# 6. BATCH COMPARISON ACROSS MULTIPLE PAIRS
# =============================================================================

def run_filter_ablation(
    pairs: List[Tuple[str, str]],
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    thresholds: Optional[HealthThresholds] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run dynamic filter ablation across multiple pairs.

    Returns a summary DataFrame comparing filtered vs unfiltered for each pair.
    """
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results'
        )
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for i, (ty, tx) in enumerate(pairs):
        pair_name = f"{ty}_{tx}"
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(pairs)}] {pair_name}")

        try:
            result = compare_filtered_vs_unfiltered(
                ty, tx, start_date, end_date, train_end_date,
                thresholds=thresholds, verbose=verbose,
            )

            mu = result['metrics_unfiltered']['oos']
            mf = result['metrics_filtered']['oos']
            fs = result['filter_stats']

            rows.append({
                'Pair': pair_name,
                'Unfiltered_Sharpe': mu['sharpe'],
                'Filtered_Sharpe': mf['sharpe'],
                'Sharpe_Change': fs['sharpe_change'],
                'Unfiltered_Return_%': mu['total_return'] * 100,
                'Filtered_Return_%': mf['total_return'] * 100,
                'Return_Change_%': fs['return_change'] * 100,
                'Unfiltered_MaxDD_%': mu['max_dd'] * 100,
                'Filtered_MaxDD_%': mf['max_dd'] * 100,
                'Unfiltered_Trades': mu['n_trades'],
                'Filtered_Trades': mf['n_trades'],
                'OOS_Pct_Healthy': fs['oos_pct_healthy'],
                'OOS_Avg_Health': fs['oos_avg_health_score'],
                'Signals_Blocked_%': fs['signals_blocked_pct'],
                'Filter_Improved': fs['sharpe_change'] > 0,
            })

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        # Save
        csv_path = os.path.join(save_dir, 'filter_ablation_summary.csv')
        summary.to_csv(csv_path, index=False)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  FILTER ABLATION SUMMARY ({len(summary)} pairs)")
            print(f"{'=' * 70}")
            print(f"\n  Pairs improved by filter: "
                  f"{summary['Filter_Improved'].sum()}/{len(summary)}")
            print(f"  Avg Sharpe change: "
                  f"{summary['Sharpe_Change'].mean():+.3f}")
            print(f"  Avg signals blocked: "
                  f"{summary['Signals_Blocked_%'].mean():.1f}%")
            print(f"  Avg OOS healthy: "
                  f"{summary['OOS_Pct_Healthy'].mean():.1f}%")

            print(f"\n  Per-pair:")
            for _, row in summary.iterrows():
                icon = "✅" if row['Filter_Improved'] else "❌"
                print(f"    {icon} {row['Pair']:.<15} "
                      f"Sharpe: {row['Unfiltered_Sharpe']:.3f} → "
                      f"{row['Filtered_Sharpe']:.3f} "
                      f"({row['Sharpe_Change']:+.3f}) "
                      f"| Healthy: {row['OOS_Pct_Healthy']:.0f}%")

            print(f"\n  Results saved to: {csv_path}")

    return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 70)
    print("  WEEK 8: DYNAMIC PAIR SELECTION FILTER")
    print("=" * 70)

    # Test on existing 3 pairs
    test_pairs = [
        ('BAC', 'PNC'),
        ('WFC', 'MS'),
        ('CVX', 'OXY'),
    ]

    # Run ablation
    summary = run_filter_ablation(
        test_pairs,
        start_date='2015-01-01',
        end_date='2025-06-30',
        train_end_date='2020-12-31',
    )

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")
