"""
Academic Baselines for Pairs Trading Comparison
=================================================
Implements 6 baseline strategies that our method must beat to demonstrate value.

Baselines:
    1. Gatev Distance Method (Gatev, Goetzmann & Rouwenhorst, 2006)
    2. OLS Cointegration (Engle-Granger with fixed hedge ratio)
    3. Kalman-Only (No Regime) — Our Kalman filter, regime always NORMAL
    4. Regime-Only (No Kalman) — OLS hedge ratio + RF regime detection
    5. Buy-and-Hold Stocks — Equal-weight portfolio of pair stocks
    6. SPY Buy-and-Hold — Market benchmark

All baselines run on the SAME data, SAME pairs, SAME train/test split,
and SAME transaction cost model for fair comparison.

Reference:
    Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006).
    "Pairs trading: Performance of a relative-value arbitrage rule."
    Review of Financial Studies, 19(3), 797-827.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller
import logging
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, ConservativeKalman, StrictRegimeClassifier,
    calculate_zscore, generate_signals, download_data
)

logger = logging.getLogger(__name__)


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def compute_metrics(returns: pd.Series, label: str = '') -> dict:
    """Compute standard performance metrics from a return series."""
    ret = returns.dropna()
    if len(ret) == 0 or ret.std() == 0:
        return {
            'Method': label, 'Total_Return_%': 0, 'Ann_Return_%': 0,
            'Ann_Vol_%': 0, 'Sharpe': 0, 'Max_DD_%': 0,
            'Win_Rate_%': 0, 'Days': 0, 'Trades': 0,
        }
    tr = (1 + ret).prod() - 1
    n = len(ret)
    ar = (1 + tr) ** (252 / n) - 1
    av = ret.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    cum = (1 + ret).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    wr = (ret > 0).mean()
    return {
        'Method': label,
        'Total_Return_%': round(tr * 100, 2),
        'Ann_Return_%': round(ar * 100, 2),
        'Ann_Vol_%': round(av * 100, 2),
        'Sharpe': round(sh, 3),
        'Max_DD_%': round(dd * 100, 2),
        'Win_Rate_%': round(wr * 100, 1),
        'Days': n,
        'Trades': 0,  # overridden by individual baselines
    }


def count_trades(signals: pd.Series) -> int:
    """Count number of round-trip trades from a signal series."""
    changes = signals.diff().abs().fillna(0)
    entries = ((changes > 0) & (signals != 0)).sum()
    return int(entries)


# =============================================================================
# BASELINE 1: GATEV DISTANCE METHOD
# =============================================================================

class GatevDistanceBaseline:
    """
    Classic distance-based pairs trading (Gatev et al., 2006).

    Formation period: Normalize prices, compute sum of squared deviations.
    Trading period: Trade when normalized spread exceeds 2σ, exit at 0.

    This uses a FIXED hedge ratio of 1 (dollar-neutral) and FIXED thresholds,
    which is the original Gatev formulation.
    """

    def __init__(self, formation_period: int = 252, entry_sigma: float = 2.0,
                 exit_sigma: float = 0.0, slippage_bps: float = 5.0):
        self.formation_period = formation_period
        self.entry_sigma = entry_sigma
        self.exit_sigma = exit_sigma
        self.slippage_bps = slippage_bps

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            train_end_date: str = '2020-12-31') -> pd.DataFrame:
        """Run Gatev distance baseline."""
        data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()
        train_end = pd.Timestamp(train_end_date)

        # Normalize prices over formation period (training window)
        train = data[data.index <= train_end]
        y_norm = data['Y'] / train['Y'].iloc[0]
        x_norm = data['X'] / train['X'].iloc[0]

        # Distance = normalized spread
        spread = y_norm - x_norm

        # Formation statistics (from training period only)
        train_spread = spread[spread.index <= train_end]
        mu = train_spread.mean()
        sigma = train_spread.std()

        if sigma == 0:
            sigma = 1e-8

        # Z-score using formation stats (frozen from train period)
        z = (spread - mu) / sigma

        # Generate signals
        signals = pd.Series(0, index=data.index)
        position = 0

        for i in range(len(z)):
            zval = z.iloc[i]
            if position == 0:
                if zval > self.entry_sigma:
                    position = -1  # short spread
                elif zval < -self.entry_sigma:
                    position = 1   # long spread
            elif position != 0:
                if abs(zval) < self.exit_sigma + 0.1:
                    position = 0
                elif (position == 1 and zval > self.entry_sigma * 2):
                    position = 0  # stop-loss
                elif (position == -1 and zval < -self.entry_sigma * 2):
                    position = 0  # stop-loss
            signals.iloc[i] = position

        # Returns (dollar-neutral: hedge ratio = 1)
        ret_y = data['Y'].pct_change()
        ret_x = data['X'].pct_change()
        spread_ret = ret_y - ret_x

        gross = signals.shift(1) * spread_ret
        cost = signals.diff().abs().fillna(0) * (self.slippage_bps / 10_000)
        net = gross - cost

        results = pd.DataFrame({
            'strategy_return': net,
            'signal': signals,
            'z_score': z,
        }, index=data.index)
        results['period'] = 'train'
        results.loc[results.index > train_end, 'period'] = 'oos'
        results['cumulative'] = (1 + results['strategy_return']).cumprod()

        return results


# =============================================================================
# BASELINE 2: OLS COINTEGRATION (ENGLE-GRANGER)
# =============================================================================

class OLSCointegrationBaseline:
    """
    Standard Engle-Granger cointegration pairs trading.

    Uses OLS regression on training data to estimate a FIXED hedge ratio,
    then trades the spread using z-score thresholds.

    This is the "textbook" approach without adaptive Kalman filtering.
    """

    def __init__(self, entry_z: float = 1.5, exit_z: float = 0.5,
                 z_window: int = 40, slippage_bps: float = 5.0):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.z_window = z_window
        self.slippage_bps = slippage_bps

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            train_end_date: str = '2020-12-31') -> pd.DataFrame:
        """Run OLS cointegration baseline."""
        data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()
        train_end = pd.Timestamp(train_end_date)

        # OLS hedge ratio estimated on training data only
        train = data[data.index <= train_end]
        model = LinearRegression()
        model.fit(train['X'].values.reshape(-1, 1), train['Y'].values)
        hedge_ratio = model.coef_[0]
        intercept = model.intercept_

        # Fixed spread
        spread = data['Y'] - hedge_ratio * data['X'] - intercept

        # Rolling z-score
        z = calculate_zscore(spread, window=self.z_window)

        # Generate signals (no regime gating — always NORMAL)
        regimes = pd.Series(2, index=data.index)
        signals = generate_signals(
            z, regimes,
            entry_z_normal=self.entry_z, exit_z_normal=self.exit_z,
            min_hold_days=3, max_hold_days=30,
        )

        # Returns
        ret_y = data['Y'].pct_change()
        ret_x = data['X'].pct_change()
        spread_ret = ret_y - hedge_ratio * ret_x

        gross = signals.shift(1) * spread_ret
        cost = signals.diff().abs().fillna(0) * (self.slippage_bps / 10_000)
        net = gross - cost

        results = pd.DataFrame({
            'strategy_return': net,
            'signal': signals,
            'z_score': z,
            'spread': spread,
            'hedge_ratio': hedge_ratio,
        }, index=data.index)
        results['period'] = 'train'
        results.loc[results.index > train_end, 'period'] = 'oos'
        results['cumulative'] = (1 + results['strategy_return']).cumprod()

        return results


# =============================================================================
# BASELINE 3: KALMAN-ONLY (NO REGIME)
# =============================================================================

class KalmanOnlyBaseline:
    """
    Our adaptive Kalman filter for hedge ratio estimation, but with
    regime detection DISABLED (always treat market as NORMAL).

    This isolates the value of the Kalman filter itself.
    """

    def __init__(self, entry_z: float = 1.5, exit_z: float = 0.5,
                 z_window: int = 40, slippage_bps: float = 5.0):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.z_window = z_window
        self.slippage_bps = slippage_bps

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            train_end_date: str = '2020-12-31') -> pd.DataFrame:
        """Run Kalman-only baseline (regime always NORMAL)."""
        data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()
        train_end = pd.Timestamp(train_end_date)

        # Kalman filter with regime always NORMAL
        kalman = ConservativeKalman()
        spreads, hedge_ratios = [], []

        for i in range(len(data)):
            y_val = data['Y'].iloc[i]
            x_val = data['X'].iloc[i]
            et, _ = kalman.update(y_val, x_val, regime='normal')
            spreads.append(et)
            hedge_ratios.append(kalman.get_hedge_ratio())

        data['spread'] = spreads
        data['hedge_ratio'] = hedge_ratios

        # Z-score
        z = calculate_zscore(
            pd.Series(spreads, index=data.index), window=self.z_window
        )

        # Signals (no regime gating)
        regimes = pd.Series(2, index=data.index)
        signals = generate_signals(
            z, regimes,
            entry_z_normal=self.entry_z, exit_z_normal=self.exit_z,
            min_hold_days=3, max_hold_days=30,
        )

        # Returns
        ret_y = data['Y'].pct_change()
        ret_x = data['X'].pct_change()
        spread_ret = ret_y - data['hedge_ratio'] * ret_x

        gross = signals.shift(1) * spread_ret
        cost = signals.diff().abs().fillna(0) * (self.slippage_bps / 10_000)
        net = gross - cost

        results = pd.DataFrame({
            'strategy_return': net,
            'signal': signals,
            'z_score': z,
        }, index=data.index)
        results['period'] = 'train'
        results.loc[results.index > train_end, 'period'] = 'oos'
        results['cumulative'] = (1 + results['strategy_return']).cumprod()

        return results


# =============================================================================
# BASELINE 4: REGIME-ONLY (NO KALMAN)
# =============================================================================

class RegimeOnlyBaseline:
    """
    OLS fixed hedge ratio + our RF regime detection.

    This isolates the value of regime detection when using a simple
    (non-adaptive) hedge ratio.
    """

    def __init__(self, entry_z: float = 1.5, exit_z: float = 0.5,
                 z_window: int = 40, slippage_bps: float = 5.0):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.z_window = z_window
        self.slippage_bps = slippage_bps

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            market_index: pd.Series,
            train_end_date: str = '2020-12-31') -> pd.DataFrame:
        """Run regime-only baseline (OLS hedge + RF regime)."""
        data = pd.DataFrame({
            'Y': stock_y, 'X': stock_x, 'Market': market_index
        }).dropna()
        train_end = pd.Timestamp(train_end_date)
        train_mask = data.index <= train_end

        # OLS hedge ratio from training data
        train = data[train_mask]
        model = LinearRegression()
        model.fit(train['X'].values.reshape(-1, 1), train['Y'].values)
        hedge_ratio = model.coef_[0]
        intercept = model.intercept_

        # Fixed spread
        spread = data['Y'] - hedge_ratio * data['X'] - intercept
        z = calculate_zscore(spread, window=self.z_window)

        # Train RF regime classifier
        classifier = StrictRegimeClassifier()
        prices_df = pd.DataFrame({'Y': data['Y'], 'X': data['X']})
        features = classifier.create_features(prices_df, data['Market'])
        labels = classifier.label_regimes(features)

        train_features = features[train_mask]
        train_labels = labels[train_mask]
        classifier.train(train_features, train_labels)
        regimes = classifier.predict(features)

        # Signals with regime gating
        signals = generate_signals(
            z, regimes,
            entry_z_normal=self.entry_z, exit_z_normal=self.exit_z,
            entry_z_volatile=self.entry_z + 0.5,
            exit_z_volatile=max(0.1, self.exit_z - 0.1),
            min_hold_days=3, max_hold_days=30,
        )

        # Returns
        ret_y = data['Y'].pct_change()
        ret_x = data['X'].pct_change()
        spread_ret = ret_y - hedge_ratio * ret_x

        gross = signals.shift(1) * spread_ret
        cost = signals.diff().abs().fillna(0) * (self.slippage_bps / 10_000)
        net = gross - cost

        results = pd.DataFrame({
            'strategy_return': net,
            'signal': signals,
            'z_score': z,
            'regime': regimes.values,
        }, index=data.index)
        results['period'] = 'train'
        results.loc[results.index > train_end, 'period'] = 'oos'
        results['cumulative'] = (1 + results['strategy_return']).cumprod()

        return results


# =============================================================================
# BASELINE 5: BUY-AND-HOLD STOCKS (EQUAL WEIGHT)
# =============================================================================

class BuyAndHoldStocksBaseline:
    """Equal-weight buy-and-hold of both pair stocks."""

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            train_end_date: str = '2020-12-31') -> pd.DataFrame:
        data = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()
        train_end = pd.Timestamp(train_end_date)

        ret_y = data['Y'].pct_change()
        ret_x = data['X'].pct_change()
        equal_weight = 0.5 * ret_y + 0.5 * ret_x

        results = pd.DataFrame({
            'strategy_return': equal_weight,
        }, index=data.index)
        results['period'] = 'train'
        results.loc[results.index > train_end, 'period'] = 'oos'
        results['cumulative'] = (1 + results['strategy_return']).cumprod()
        return results


# =============================================================================
# BASELINE 6: SPY BUY-AND-HOLD
# =============================================================================

class SPYBuyAndHoldBaseline:
    """Market benchmark — SPY buy and hold."""

    def run(self, market_index: pd.Series,
            train_end_date: str = '2020-12-31') -> pd.DataFrame:
        data = pd.DataFrame({'Market': market_index}).dropna()
        train_end = pd.Timestamp(train_end_date)

        results = pd.DataFrame({
            'strategy_return': data['Market'].pct_change(),
        }, index=data.index)
        results['period'] = 'train'
        results.loc[results.index > train_end, 'period'] = 'oos'
        results['cumulative'] = (1 + results['strategy_return']).cumprod()
        return results


# =============================================================================
# UNIFIED COMPARISON RUNNER
# =============================================================================

def run_all_baselines(ticker_y: str, ticker_x: str,
                      start_date: str = '2015-01-01',
                      end_date: str = '2025-06-30',
                      train_end_date: str = '2020-12-31',
                      slippage_bps: float = 5.0,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Run all 6 baselines + our full system on a single pair.
    Returns a DataFrame comparing all methods on OOS period.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  BASELINE COMPARISON: {ticker_y}/{ticker_x}")
        print(f"  Train ≤ {train_end_date} | Slippage: {slippage_bps}bps")
        print(f"{'=' * 70}")

    # Download data
    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    all_results = {}
    comparison = []

    # --- Our Full System ---
    if verbose:
        print("\n  [1/7] Our Full System (Kalman + Regime)...")
    try:
        system = ConservativeSystem()
        our = system.run_backtest(
            stock_y, stock_x, market,
            train_end_date=train_end_date,
            slippage_bps=slippage_bps,
            verbose=False,
        )
        oos = our[our['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'Ours (Kalman+Regime)')
        trades = system.get_trade_analysis(period='oos')
        m['Trades'] = len(trades)
        comparison.append(m)
        all_results['full_system'] = {
            'data': our,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"Our system failed: {e}")

    # --- Baseline 1: Gatev Distance ---
    if verbose:
        print("  [2/7] Gatev Distance Method...")
    try:
        gatev = GatevDistanceBaseline(slippage_bps=slippage_bps)
        res = gatev.run(stock_y, stock_x, train_end_date)
        oos = res[res['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'Gatev Distance')
        m['Trades'] = count_trades(oos['signal'])
        comparison.append(m)
        all_results['gatev'] = {
            'data': res,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"Gatev failed: {e}")

    # --- Baseline 2: OLS Cointegration ---
    if verbose:
        print("  [3/7] OLS Cointegration (Engle-Granger)...")
    try:
        ols = OLSCointegrationBaseline(slippage_bps=slippage_bps)
        res = ols.run(stock_y, stock_x, train_end_date)
        oos = res[res['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'OLS Cointegration')
        m['Trades'] = count_trades(oos['signal'])
        comparison.append(m)
        all_results['ols'] = {
            'data': res,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"OLS failed: {e}")

    # --- Baseline 3: Kalman-Only ---
    if verbose:
        print("  [4/7] Kalman-Only (No Regime)...")
    try:
        ko = KalmanOnlyBaseline(slippage_bps=slippage_bps)
        res = ko.run(stock_y, stock_x, train_end_date)
        oos = res[res['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'Kalman-Only (No Regime)')
        m['Trades'] = count_trades(oos['signal'])
        comparison.append(m)
        all_results['kalman_only'] = {
            'data': res,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"Kalman-only failed: {e}")

    # --- Baseline 4: Regime-Only ---
    if verbose:
        print("  [5/7] Regime-Only (No Kalman)...")
    try:
        ro = RegimeOnlyBaseline(slippage_bps=slippage_bps)
        res = ro.run(stock_y, stock_x, market, train_end_date)
        oos = res[res['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'Regime-Only (No Kalman)')
        m['Trades'] = count_trades(oos['signal'])
        comparison.append(m)
        all_results['regime_only'] = {
            'data': res,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"Regime-only failed: {e}")

    # --- Baseline 5: B&H Stocks ---
    if verbose:
        print("  [6/7] Buy-and-Hold Stocks (Equal Weight)...")
    try:
        bh = BuyAndHoldStocksBaseline()
        res = bh.run(stock_y, stock_x, train_end_date)
        oos = res[res['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'B&H Stocks (50/50)')
        m['Trades'] = 0
        comparison.append(m)
        all_results['bh_stocks'] = {
            'data': res,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"B&H Stocks failed: {e}")

    # --- Baseline 6: SPY B&H ---
    if verbose:
        print("  [7/7] SPY Buy-and-Hold...")
    try:
        spy = SPYBuyAndHoldBaseline()
        res = spy.run(market, train_end_date)
        oos = res[res['period'] == 'oos']
        m = compute_metrics(oos['strategy_return'], 'SPY Buy-and-Hold')
        m['Trades'] = 0
        comparison.append(m)
        all_results['spy'] = {
            'data': res,
            'oos_returns': oos['strategy_return'].dropna(),
        }
    except Exception as e:
        logger.warning(f"SPY failed: {e}")

    comparison_df = pd.DataFrame(comparison)

    if verbose and len(comparison_df) > 0:
        print(f"\n{'=' * 70}")
        print(f"  OOS COMPARISON TABLE ({ticker_y}/{ticker_x})")
        print(f"{'=' * 70}")
        display_cols = ['Method', 'Total_Return_%', 'Sharpe', 'Max_DD_%',
                        'Win_Rate_%', 'Trades']
        print(comparison_df[display_cols].to_string(index=False))

        best = comparison_df.loc[comparison_df['Sharpe'].idxmax()]
        print(f"\n  Best OOS Sharpe: {best['Method']} ({best['Sharpe']:.3f})")

    return comparison_df, all_results


def run_baselines_multi_pair(pairs: list,
                             start_date: str = '2015-01-01',
                             end_date: str = '2025-06-30',
                             train_end_date: str = '2020-12-31',
                             slippage_bps: float = 5.0) -> pd.DataFrame:
    """
    Run all baselines across multiple pairs and aggregate.

    Parameters
    ----------
    pairs : list of (ticker_y, ticker_x) tuples
    """
    all_comparisons = []

    for i, (ty, tx) in enumerate(pairs):
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{len(pairs)}] {ty}/{tx}")
        try:
            comp_df, _ = run_all_baselines(
                ty, tx, start_date, end_date, train_end_date,
                slippage_bps, verbose=False,
            )
            comp_df['Pair'] = f"{ty}_{tx}"
            all_comparisons.append(comp_df)
            # Quick summary
            ours_row = comp_df[comp_df['Method'].str.contains('Ours')]
            if len(ours_row) > 0:
                sh = ours_row['Sharpe'].values[0]
                best_base = comp_df[~comp_df['Method'].str.contains('Ours')]['Sharpe'].max()
                print(f"    Ours Sharpe={sh:.3f} vs Best Baseline={best_base:.3f} "
                      f"{'✅' if sh > best_base else '❌'}")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    if len(all_comparisons) == 0:
        return pd.DataFrame()

    full = pd.concat(all_comparisons, ignore_index=True)

    # Aggregate by method
    agg = full.groupby('Method').agg({
        'Sharpe': ['mean', 'median', 'std'],
        'Total_Return_%': ['mean', 'median'],
        'Max_DD_%': 'mean',
        'Win_Rate_%': 'mean',
    }).round(3)

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE BASELINE COMPARISON ({len(pairs)} pairs)")
    print(f"{'=' * 70}")
    print(agg.to_string())

    # How often does ours beat each baseline?
    print(f"\n  Win Rates (Ours vs Each Baseline):")
    ours_per_pair = full[full['Method'].str.contains('Ours')].set_index('Pair')['Sharpe']
    for method in full['Method'].unique():
        if 'Ours' in method:
            continue
        base_per_pair = full[full['Method'] == method].set_index('Pair')['Sharpe']
        common = ours_per_pair.index.intersection(base_per_pair.index)
        if len(common) > 0:
            wins = (ours_per_pair[common] > base_per_pair[common]).mean()
            print(f"    vs {method:.<35} {wins*100:.0f}% wins")

    return full


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Single pair demo
    comp_df, all_res = run_all_baselines('BAC', 'PNC')

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    comp_df.to_csv(os.path.join(results_dir, 'baseline_comparison_BAC_PNC.csv'),
                   index=False)
    print(f"\n  Saved to Research/results/baseline_comparison_BAC_PNC.csv")
