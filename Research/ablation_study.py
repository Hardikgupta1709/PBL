import numpy as np
import pandas as pd
import os
import sys
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeKalman, StrictRegimeClassifier, ConservativeSystem,
    calculate_zscore, generate_signals, download_data
)
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED METRIC COMPUTATION
# =============================================================================

def compute_oos_metrics(results: pd.DataFrame) -> dict:
    """Extract OOS metrics from a results DataFrame with 'period' column."""
    oos = results[results['period'] == 'oos']
    ret = oos['strategy_return'].dropna()
    if len(ret) == 0 or ret.std() == 0:
        return {'Sharpe': 0, 'Total_Return_%': 0, 'Max_DD_%': 0,
                'Win_Rate_%': 0, 'Trades': 0, 'Ann_Vol_%': 0}
    tr = (1 + ret).prod() - 1
    n = len(ret)
    ar = (1 + tr) ** (252 / n) - 1
    av = ret.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    cum = (1 + ret).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    trades = (oos.get('final_signal', oos.get('signal', pd.Series(0))).diff().abs() > 0).sum() // 2
    return {
        'Sharpe': round(sh, 4),
        'Total_Return_%': round(tr * 100, 2),
        'Ann_Return_%': round(ar * 100, 2),
        'Ann_Vol_%': round(av * 100, 2),
        'Max_DD_%': round(dd * 100, 2),
        'Win_Rate_%': round((ret > 0).mean() * 100, 1),
        'Trades': int(trades),
    }


# =============================================================================
# ABLATION 0: FULL SYSTEM (REFERENCE)
# =============================================================================

def run_full_system(stock_y, stock_x, market, train_end_date,
                    slippage_bps=5.0) -> pd.DataFrame:
    """Full system — all components enabled."""
    system = ConservativeSystem()
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=2.0, exit_z_volatile=0.3,
        min_hold_days=3, max_hold_days=30,
        stop_loss_mult=2.0, z_score_window=40,
        slippage_bps=slippage_bps, verbose=False,
    )


# =============================================================================
# ABLATION 1: REMOVE MAD OUTLIER DETECTION
# =============================================================================

class KalmanNoMAD(ConservativeKalman):
    """Kalman filter with MAD outlier detection disabled."""

    def update(self, y, x, regime='normal'):
        if self.wt is None:
            self.initialize(y, x)
            return 0.0, np.sqrt(self.var_e)

        noise_multiplier = {'crisis': 5.0, 'volatile': 2.0, 'normal': 1.0}
        var_eta_adjusted = self.var_eta * noise_multiplier.get(regime, 1.0)

        wt_minus = self.wt
        self.At = self.Ct + var_eta_adjusted
        et = y - wt_minus * x
        self.Rt = self.At * x**2 + self.var_e

        # Standard Kalman gain — NO MAD check
        Kt = self.At * x / self.Rt if self.Rt != 0 else 0

        self.wt = wt_minus + Kt * et
        self.Ct = self.At - Kt * x * self.At
        return et, np.sqrt(self.Rt)


def run_ablation_no_mad(stock_y, stock_x, market, train_end_date,
                        slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 1: Remove MAD outlier detection."""
    system = ConservativeSystem()
    system.kalman = KalmanNoMAD()  # Replace with non-MAD Kalman
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        slippage_bps=slippage_bps, verbose=False,
    )


# =============================================================================
# ABLATION 2: REMOVE REGIME CLASSIFICATION (ALWAYS NORMAL)
# =============================================================================

def run_ablation_no_regime(stock_y, stock_x, market, train_end_date,
                           slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 2: Disable regime — treat all days as NORMAL."""
    data = pd.DataFrame({
        'Y': stock_y, 'X': stock_x, 'Market': market
    }).dropna()
    train_end = pd.Timestamp(train_end_date)

    # Kalman filter
    kalman = ConservativeKalman()
    spreads, hedge_ratios = [], []
    for i in range(len(data)):
        et, _ = kalman.update(data['Y'].iloc[i], data['X'].iloc[i], regime='normal')
        spreads.append(et)
        hedge_ratios.append(kalman.get_hedge_ratio())

    data['spread'] = spreads
    data['hedge_ratio'] = hedge_ratios
    data['z_score'] = calculate_zscore(pd.Series(spreads, index=data.index), 40)

    # All regimes set to NORMAL (2)
    regimes = pd.Series(2, index=data.index)
    data['regime'] = regimes
    data['final_signal'] = generate_signals(
        data['z_score'], regimes,
        entry_z_normal=1.5, exit_z_normal=0.5,
        min_hold_days=3, max_hold_days=30, stop_loss_mult=2.0,
    )

    data['returns_Y'] = data['Y'].pct_change()
    data['returns_X'] = data['X'].pct_change()
    data['spread_return'] = data['returns_Y'] - data['hedge_ratio'] * data['returns_X']
    data['strategy_return_gross'] = data['final_signal'].shift(1) * data['spread_return']
    cost = data['final_signal'].diff().abs().fillna(0) * (slippage_bps / 10_000)
    data['strategy_return'] = data['strategy_return_gross'] - cost

    data['period'] = 'train'
    data.loc[data.index > train_end, 'period'] = 'oos'
    data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
    return data


# =============================================================================
# ABLATION 3: REMOVE ADAPTIVE KALMAN (USE FIXED OLS HEDGE)
# =============================================================================

def run_ablation_no_kalman(stock_y, stock_x, market, train_end_date,
                           slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 3: Use fixed OLS hedge ratio instead of adaptive Kalman."""
    data = pd.DataFrame({
        'Y': stock_y, 'X': stock_x, 'Market': market
    }).dropna()
    train_end = pd.Timestamp(train_end_date)
    train_mask = data.index <= train_end

    # OLS hedge from training data
    train = data[train_mask]
    model = LinearRegression()
    model.fit(train['X'].values.reshape(-1, 1), train['Y'].values)
    hedge_ratio = model.coef_[0]
    intercept = model.intercept_

    spread = data['Y'] - hedge_ratio * data['X'] - intercept
    data['spread'] = spread
    data['hedge_ratio'] = hedge_ratio
    data['z_score'] = calculate_zscore(spread, 40)

    # Still use regime classification
    classifier = StrictRegimeClassifier()
    prices_df = pd.DataFrame({'Y': data['Y'], 'X': data['X']})
    features = classifier.create_features(prices_df, data['Market'])
    labels = classifier.label_regimes(features)
    classifier.train(features[train_mask], labels[train_mask])
    data['regime'] = classifier.predict(features)

    data['final_signal'] = generate_signals(
        data['z_score'], data['regime'],
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=2.0, exit_z_volatile=0.3,
        min_hold_days=3, max_hold_days=30, stop_loss_mult=2.0,
    )

    data['returns_Y'] = data['Y'].pct_change()
    data['returns_X'] = data['X'].pct_change()
    data['spread_return'] = data['returns_Y'] - hedge_ratio * data['returns_X']
    data['strategy_return_gross'] = data['final_signal'].shift(1) * data['spread_return']
    cost = data['final_signal'].diff().abs().fillna(0) * (slippage_bps / 10_000)
    data['strategy_return'] = data['strategy_return_gross'] - cost

    data['period'] = 'train'
    data.loc[data.index > train_end, 'period'] = 'oos'
    data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
    return data


# =============================================================================
# ABLATION 4: FIXED Z-SCORE WINDOW (REMOVE ADAPTIVE HALF-LIFE)
# =============================================================================

def run_ablation_fixed_window(stock_y, stock_x, market, train_end_date,
                              fixed_window: int = 20,
                              slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 4: Use a different fixed z-score window (20 instead of 40)."""
    system = ConservativeSystem()
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        z_score_window=fixed_window,
        slippage_bps=slippage_bps, verbose=False,
    )


# =============================================================================
# ABLATION 5: REMOVE STOP-LOSS
# =============================================================================

def run_ablation_no_stoploss(stock_y, stock_x, market, train_end_date,
                             slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 5: Remove stop-loss (set multiplier very high)."""
    system = ConservativeSystem()
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        stop_loss_mult=100.0,  # effectively no stop-loss
        slippage_bps=slippage_bps, verbose=False,
    )


# =============================================================================
# ABLATION 6: REMOVE MINIMUM HOLD PERIOD
# =============================================================================

def run_ablation_no_min_hold(stock_y, stock_x, market, train_end_date,
                             slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 6: Remove min hold period (set to 0)."""
    system = ConservativeSystem()
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        min_hold_days=0,
        slippage_bps=slippage_bps, verbose=False,
    )


# =============================================================================
# ABLATION 7: REMOVE TRANSACTION COSTS (GROSS RETURNS)
# =============================================================================

def run_ablation_no_costs(stock_y, stock_x, market, train_end_date) -> pd.DataFrame:
    """Ablation 7: Zero transaction costs — shows gross vs net gap."""
    system = ConservativeSystem()
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        slippage_bps=0.0,
        verbose=False,
    )


# =============================================================================
# ABLATION 8: REMOVE VOLATILE-REGIME PARAMETER ADJUSTMENT
# =============================================================================

def run_ablation_no_volatile_adjust(stock_y, stock_x, market, train_end_date,
                                    slippage_bps=5.0) -> pd.DataFrame:
    """Ablation 8: Use same params for NORMAL and VOLATILE (no 2-tier)."""
    system = ConservativeSystem()
    return system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=1.5,   # same as normal
        exit_z_volatile=0.5,    # same as normal
        slippage_bps=slippage_bps, verbose=False,
    )


# =============================================================================
# ABLATION RUNNER
# =============================================================================

ABLATION_CONFIGS = {
    'Full System': run_full_system,
    '−MAD Outlier Detection': run_ablation_no_mad,
    '−Regime Classification': run_ablation_no_regime,
    '−Adaptive Kalman (→OLS)': run_ablation_no_kalman,
    '−Z-Score Window (40→20)': lambda y, x, m, t, s=5.0: run_ablation_fixed_window(y, x, m, t, 20, s),
    '−Stop-Loss': run_ablation_no_stoploss,
    '−Minimum Hold Period': run_ablation_no_min_hold,
    '−Transaction Costs': lambda y, x, m, t, s=5.0: run_ablation_no_costs(y, x, m, t),
    '−Volatile Param Adjust': run_ablation_no_volatile_adjust,
}


def run_ablation_single_pair(ticker_y: str, ticker_x: str,
                             start_date: str = '2015-01-01',
                             end_date: str = '2025-06-30',
                             train_end_date: str = '2020-12-31',
                             slippage_bps: float = 5.0,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Run all ablation configs on a single pair.
    Returns DataFrame with one row per config.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  ABLATION STUDY: {ticker_y}/{ticker_x}")
        print(f"  Train ≤ {train_end_date}")
        print(f"{'=' * 70}")

    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    rows = []
    ref_sharpe = None

    for name, run_fn in ABLATION_CONFIGS.items():
        try:
            results = run_fn(stock_y, stock_x, market, train_end_date, slippage_bps)
            metrics = compute_oos_metrics(results)
            metrics['Config'] = name

            if ref_sharpe is None:
                ref_sharpe = metrics['Sharpe']
                metrics['Sharpe_Delta'] = 0.0
            else:
                metrics['Sharpe_Delta'] = round(metrics['Sharpe'] - ref_sharpe, 4)

            rows.append(metrics)

            if verbose:
                delta_str = f"Δ={metrics['Sharpe_Delta']:+.4f}" if metrics['Sharpe_Delta'] != 0 else "REFERENCE"
                icon = "📊" if 'Full' in name else ("🔴" if metrics['Sharpe_Delta'] < -0.05 else ("🟢" if metrics['Sharpe_Delta'] > 0.05 else "⚪"))
                print(f"  {icon} {name:.<40} Sharpe={metrics['Sharpe']:.4f}  {delta_str}")

        except Exception as e:
            logger.warning(f"Ablation '{name}' failed: {e}")
            rows.append({'Config': name, 'Sharpe': np.nan, 'Sharpe_Delta': np.nan})

    df = pd.DataFrame(rows)

    if verbose and len(df) > 0:
        print(f"\n  Components ranked by impact (removing hurts most → most valuable):")
        ranked = df[df['Config'] != 'Full System'].dropna(subset=['Sharpe_Delta'])
        ranked = ranked.sort_values('Sharpe_Delta')
        for _, row in ranked.iterrows():
            direction = "📉" if row['Sharpe_Delta'] < 0 else "📈"
            print(f"    {direction} {row['Config']:.<40} {row['Sharpe_Delta']:+.4f}")

    return df


def run_ablation_multi_pair(pairs: list,
                            start_date: str = '2015-01-01',
                            end_date: str = '2025-06-30',
                            train_end_date: str = '2020-12-31',
                            slippage_bps: float = 5.0) -> pd.DataFrame:
    """
    Run ablation across multiple pairs and aggregate.
    """
    all_results = []

    for i, (ty, tx) in enumerate(pairs):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(pairs)}] Ablation: {ty}/{tx}")
        try:
            df = run_ablation_single_pair(
                ty, tx, start_date, end_date, train_end_date,
                slippage_bps, verbose=False,
            )
            df['Pair'] = f"{ty}_{tx}"
            all_results.append(df)

            # Quick summary
            full = df[df['Config'] == 'Full System']['Sharpe'].values[0]
            worst_ablation = df[df['Config'] != 'Full System']['Sharpe_Delta'].min()
            worst_name = df.loc[df['Sharpe_Delta'] == worst_ablation, 'Config'].values
            worst_name = worst_name[0] if len(worst_name) > 0 else 'N/A'
            print(f"    Full Sharpe={full:.4f} | Most valuable: {worst_name} (Δ={worst_ablation:+.4f})")

        except Exception as e:
            print(f"    ✗ Error: {e}")

    if len(all_results) == 0:
        return pd.DataFrame()

    full_df = pd.concat(all_results, ignore_index=True)

    # Aggregate: mean Sharpe delta per config across all pairs
    agg = full_df.groupby('Config').agg({
        'Sharpe': ['mean', 'median', 'std'],
        'Sharpe_Delta': ['mean', 'median'],
        'Total_Return_%': 'mean',
    }).round(4)

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE ABLATION RESULTS ({len(pairs)} pairs)")
    print(f"{'=' * 70}")
    print(agg.to_string())

    # Rank components by importance
    avg_delta = full_df[full_df['Config'] != 'Full System'].groupby('Config')['Sharpe_Delta'].mean()
    avg_delta = avg_delta.sort_values()

    print(f"\n  COMPONENT RANKING (by avg Sharpe impact when removed):")
    for config, delta in avg_delta.items():
        bar_len = max(1, int(abs(delta) * 200))
        direction = "📉" if delta < 0 else "📈"
        bar = "█" * min(bar_len, 30)
        print(f"    {direction} {config:.<40} {delta:+.4f}  {bar}")

    return full_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Single pair demo
    df = run_ablation_single_pair('BAC', 'PNC')

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'ablation_BAC_PNC.csv'), index=False)
    print(f"\n  Saved to Research/results/ablation_BAC_PNC.csv")
