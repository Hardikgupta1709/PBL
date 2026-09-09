"""
Strategy Validator: Academic-Grade Validation Suite
====================================================
Implements all validation tests required for conference-level research:

1. Expanding Walk-Forward Analysis (10+ folds, quarterly steps)
2. Monte Carlo Simulation (random entry/exit dates, NOT shuffled returns)
3. Bootstrap Confidence Intervals for Sharpe ratio
4. Stress Testing across historical crises
5. Sensitivity Analysis 
6. Statistical Tests (stationarity, mean, autocorrelation)
7. Benjamini-Hochberg multiple testing correction

References:
    - Harvey et al. (2016) "...and the Cross-Section of Expected Returns"
    - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
    - White (2000) "A Reality Check for Data Snooping"
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import warnings
import logging
import yaml
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, download_data, calculate_zscore,
    calculate_half_life, generate_signals
)


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def load_config(config_path: str = None) -> dict:
    """Load experiment configuration from YAML."""
    if config_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base, 'Research', 'config_experiments.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# =============================================================================
# 1. EXPANDING WALK-FORWARD ANALYSIS
# =============================================================================

class WalkForwardAnalyzer:
    """
    Expanding-window walk-forward analysis.

    Unlike the old sliding-window approach, this uses an EXPANDING training
    window: each fold adds more history. This mirrors real deployment where
    you never throw away data.

    Protocol:
        Fold 1: Train [0, T₀], Test [T₀, T₀ + step]
        Fold 2: Train [0, T₀ + step], Test [T₀ + step, T₀ + 2*step]
        ...and so on.
    """

    def __init__(self, min_train_days: int = 504, step_days: int = 63,
                 test_days: int = 126, min_folds: int = 8):
        self.min_train_days = min_train_days
        self.step_days = step_days
        self.test_days = test_days
        self.min_folds = min_folds
        self.fold_results = []

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            market_index: pd.Series,
            entry_z_normal: float = 1.5, exit_z_normal: float = 0.5,
            entry_z_volatile: float = 2.0, exit_z_volatile: float = 0.3,
            z_score_window: int = 40, slippage_bps: float = 5.0,
            verbose: bool = True) -> pd.DataFrame:
        """Run expanding walk-forward analysis."""

        # Align data
        df = pd.DataFrame({
            'Y': stock_y, 'X': stock_x, 'M': market_index
        }).dropna()
        n = len(df)

        if verbose:
            print("\n" + "=" * 70)
            print("  1. EXPANDING WALK-FORWARD ANALYSIS")
            print("=" * 70)
            print(f"  Min train: {self.min_train_days}d | Step: {self.step_days}d | "
                  f"Test: {self.test_days}d")

        self.fold_results = []
        fold_num = 0

        # First train window ends at min_train_days
        train_end_idx = self.min_train_days

        while train_end_idx + self.test_days <= n:
            fold_num += 1
            test_end_idx = min(train_end_idx + self.test_days, n)

            train_end_date = df.index[train_end_idx - 1].strftime('%Y-%m-%d')

            fold_y = df['Y'].iloc[:test_end_idx]
            fold_x = df['X'].iloc[:test_end_idx]
            fold_m = df['M'].iloc[:test_end_idx]

            try:
                system = ConservativeSystem()
                results = system.run_backtest(
                    fold_y, fold_x, fold_m,
                    train_end_date=train_end_date,
                    entry_z_normal=entry_z_normal,
                    exit_z_normal=exit_z_normal,
                    entry_z_volatile=entry_z_volatile,
                    exit_z_volatile=exit_z_volatile,
                    z_score_window=z_score_window,
                    slippage_bps=slippage_bps,
                    verbose=False
                )

                oos_data = results[results['period'] == 'oos']
                if len(oos_data) < 20:
                    train_end_idx += self.step_days
                    continue

                oos_ret = oos_data['strategy_return'].dropna()
                total_ret = (1 + oos_ret).prod() - 1
                ann_ret = (1 + total_ret) ** (252 / len(oos_ret)) - 1
                ann_vol = oos_ret.std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                cum = (1 + oos_ret).cumprod()
                max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()

                self.fold_results.append({
                    'Fold': fold_num,
                    'Train_Start': df.index[0].date(),
                    'Train_End': df.index[train_end_idx - 1].date(),
                    'Test_Start': df.index[train_end_idx].date(),
                    'Test_End': df.index[test_end_idx - 1].date(),
                    'Train_Days': train_end_idx,
                    'Test_Days': test_end_idx - train_end_idx,
                    'Return_%': total_ret * 100,
                    'Ann_Return_%': ann_ret * 100,
                    'Sharpe': sharpe,
                    'Max_DD_%': max_dd * 100,
                    'Win_Rate_%': (oos_ret > 0).mean() * 100,
                })

                if verbose:
                    print(f"  Fold {fold_num}: Train→{train_end_date} "
                          f"| OOS Ret={total_ret*100:+.2f}% "
                          f"| Sharpe={sharpe:.2f}")

            except Exception as e:
                logger.warning(f"Walk-forward fold {fold_num} failed: {e}")

            train_end_idx += self.step_days

        results_df = pd.DataFrame(self.fold_results)

        if verbose and len(results_df) > 0:
            print(f"\n  Folds Completed: {len(results_df)}")
            print(f"  Avg OOS Return: {results_df['Return_%'].mean():.2f}%")
            print(f"  Avg OOS Sharpe: {results_df['Sharpe'].mean():.3f}")
            print(f"  Consistency (>0): {(results_df['Return_%'] > 0).mean()*100:.0f}%")

            # --- Trend Analysis (is performance decaying?) ---
            self._trend_analysis(results_df, verbose)

            self._assess(results_df)

        return results_df

    @staticmethod
    def _trend_analysis(df: pd.DataFrame, verbose: bool = True) -> dict:
        """Spearman rank correlation of fold # vs Sharpe/Return to detect decay."""
        if len(df) < 4:
            if verbose:
                print("  ⚠️ Too few folds for trend analysis")
            return {}

        from scipy.stats import spearmanr

        fold_nums = np.arange(len(df))
        sharpe_corr, sharpe_pval = spearmanr(fold_nums, df['Sharpe'].values)
        ret_corr, ret_pval = spearmanr(fold_nums, df['Return_%'].values)

        trend = {
            'sharpe_trend_rho': round(sharpe_corr, 4),
            'sharpe_trend_pval': round(sharpe_pval, 4),
            'return_trend_rho': round(ret_corr, 4),
            'return_trend_pval': round(ret_pval, 4),
        }

        if verbose:
            direction_s = "improving" if sharpe_corr > 0 else "decaying"
            sig_s = "significant" if sharpe_pval < 0.05 else "not significant"
            print(f"\n  Trend Analysis (Spearman):")
            print(f"    Sharpe vs Fold#: ρ={sharpe_corr:+.3f}, p={sharpe_pval:.3f} "
                  f"→ {direction_s} ({sig_s})")
            print(f"    Return vs Fold#: ρ={ret_corr:+.3f}, p={ret_pval:.3f}")
            if sharpe_corr < -0.5 and sharpe_pval < 0.1:
                print("    ⚠️ WARNING: Performance appears to be decaying over time")
            elif sharpe_corr > 0.3:
                print("    ✅ Performance is stable or improving")

        return trend

    @staticmethod
    def diebold_mariano_test(returns_1: pd.Series, returns_2: pd.Series,
                             h: int = 1, verbose: bool = True) -> dict:
        """
        Diebold-Mariano test for comparing forecast accuracy.

        Tests H₀: E[d_t] = 0 where d_t = e1_t² − e2_t²
        For return comparison: d_t = r1_t − r2_t (loss differential).

        Uses Newey-West HAC standard errors for h-step-ahead.

        Reference:
            Diebold & Mariano (1995) "Comparing Predictive Accuracy"
        """
        r1 = returns_1.dropna().values
        r2 = returns_2.dropna().values
        n = min(len(r1), len(r2))
        r1, r2 = r1[:n], r2[:n]

        # Loss differential (squared loss)
        d = r1 ** 2 - r2 ** 2
        d_bar = d.mean()

        if n < 10:
            return {'dm_stat': 0, 'p_value': 1.0, 'n': n, 'conclusion': 'insufficient data'}

        # HAC variance (Newey-West)
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0
            gamma_sum += 2 * gamma_k

        var_d = (gamma_0 + gamma_sum) / n
        var_d = max(var_d, 1e-12)

        dm_stat = d_bar / np.sqrt(var_d)
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))

        if p_value < 0.05:
            conclusion = 'significantly different' if dm_stat < 0 else 'significantly worse'
        else:
            conclusion = 'no significant difference'

        result = {
            'dm_stat': round(dm_stat, 4),
            'p_value': round(p_value, 4),
            'd_bar': round(d_bar, 6),
            'n': n,
            'conclusion': conclusion,
        }

        if verbose:
            print(f"  Diebold-Mariano: DM={dm_stat:.3f}, p={p_value:.4f} → {conclusion}")

        return result

    @staticmethod
    def _assess(df: pd.DataFrame) -> None:
        avg_ret = df['Return_%'].mean()
        consistency = (df['Return_%'] > 0).mean()
        avg_sharpe = df['Sharpe'].mean()
        if avg_ret > 0 and consistency >= 0.6 and avg_sharpe > 0.3:
            print("  ✅ PASS: Consistent OOS performance")
        elif avg_ret > 0 and consistency >= 0.5:
            print("  ⚠️ MARGINAL: Some OOS performance")
        else:
            print("  ❌ FAIL: Poor OOS performance — likely overfit")


# =============================================================================
# 2. MONTE CARLO (RANDOM ENTRY/EXIT DATES)
# =============================================================================

class MonteCarloValidator:
    """
    Monte Carlo significance test using random entry/exit dates.

    CRITICAL FIX: The old Monte Carlo shuffled trade returns, which is
    statistically meaningless (the mean is preserved by permutation).

    New protocol:
        1. Take the actual spread series from the backtest.
        2. Generate N random strategies by picking random entry dates and
           holding for random durations drawn from the empirical distribution.
        3. Compute Sharpe ratio for each random strategy.
        4. p-value = P(random_Sharpe ≥ actual_Sharpe).

    This tests whether the TIMING of entries matters, which is the real
    question for a mean-reversion strategy.
    """

    def __init__(self, n_simulations: int = 10000,
                 confidence_level: float = 0.95, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_seed)

    def run(self, system: ConservativeSystem,
            period: str = 'oos', verbose: bool = True) -> dict:
        """Run Monte Carlo with randomized entry/exit dates."""

        if system.results is None:
            raise ValueError("Must run backtest first")

        data = system.results.copy()
        if period == 'oos':
            data = data[data['period'] == 'oos']
        elif period == 'train':
            data = data[data['period'] == 'train']

        if verbose:
            print("\n" + "=" * 70)
            print("  2. MONTE CARLO SIMULATION (Random Entry/Exit Dates)")
            print("=" * 70)

        # Extract actual trade statistics
        trades = system.get_trade_analysis(period=period)
        if len(trades) < 3:
            if verbose:
                print("  ⚠️ Fewer than 3 trades — cannot run Monte Carlo")
            return {'p_value_sharpe': 1.0, 'p_value_return': 1.0,
                    'actual_sharpe': 0, 'n_sims': 0}

        n_trades = len(trades)
        durations = trades['duration_days'].values
        mean_duration = max(int(np.mean(durations)), 2)
        std_duration = max(int(np.std(durations)), 1)

        # Actual strategy metrics on this period
        actual_ret = data['strategy_return'].dropna()
        actual_total = (1 + actual_ret).prod() - 1
        actual_vol = actual_ret.std() * np.sqrt(252)
        actual_ann = (1 + actual_total) ** (252 / len(actual_ret)) - 1
        actual_sharpe = actual_ann / actual_vol if actual_vol > 0 else 0

        # Spread return series (the raw material)
        spread_ret = data['spread_return'].dropna().values
        n_days = len(spread_ret)

        if verbose:
            print(f"  Actual Trades: {n_trades} | Actual Sharpe: {actual_sharpe:.3f}")
            print(f"  Actual Total Return: {actual_total*100:.2f}%")
            print(f"  Mean holding: {mean_duration}d | Running {self.n_simulations} sims...")

        sim_sharpes = np.zeros(self.n_simulations)
        sim_returns = np.zeros(self.n_simulations)

        for s in range(self.n_simulations):
            # Generate random entry/exit schedule
            sim_signal = np.zeros(n_days)
            i = 0
            trades_placed = 0

            while trades_placed < n_trades and i < n_days - 2:
                # Random gap before next entry
                gap = self.rng.geometric(p=n_trades / n_days)
                i += gap
                if i >= n_days - 2:
                    break

                # Random holding period from empirical distribution
                hold = max(2, int(self.rng.normal(mean_duration, std_duration)))
                hold = min(hold, n_days - i - 1)

                # Random direction (long or short spread)
                direction = self.rng.choice([-1, 1])
                sim_signal[i:i + hold] = direction
                i += hold
                trades_placed += 1

            # Compute returns
            sim_daily_ret = np.roll(sim_signal, 1) * spread_ret
            sim_daily_ret[0] = 0
            total = np.prod(1 + sim_daily_ret) - 1
            vol = np.std(sim_daily_ret) * np.sqrt(252)
            ann = (1 + total) ** (252 / n_days) - 1 if total > -1 else -1
            sim_sharpes[s] = ann / vol if vol > 0 else 0
            sim_returns[s] = total

        # p-values
        p_value_sharpe = (sim_sharpes >= actual_sharpe).mean()
        p_value_return = (sim_returns >= actual_total).mean()

        # Confidence interval of simulated distribution
        ci_lo = np.percentile(sim_sharpes, (1 - self.confidence_level) / 2 * 100)
        ci_hi = np.percentile(sim_sharpes, (1 + self.confidence_level) / 2 * 100)

        results = {
            'actual_sharpe': actual_sharpe,
            'actual_total_return': actual_total,
            'sim_sharpes': sim_sharpes,
            'sim_returns': sim_returns,
            'p_value_sharpe': p_value_sharpe,
            'p_value_return': p_value_return,
            'sharpe_ci_lower': ci_lo,
            'sharpe_ci_upper': ci_hi,
            'n_sims': self.n_simulations,
            'n_trades': n_trades,
        }

        if verbose:
            print(f"\n  Simulated Sharpe: mean={sim_sharpes.mean():.3f} "
                  f"std={sim_sharpes.std():.3f}")
            print(f"  95% CI (null): [{ci_lo:.3f}, {ci_hi:.3f}]")
            print(f"  p-value (Sharpe): {p_value_sharpe:.4f}")
            print(f"  p-value (Return): {p_value_return:.4f}")
            if p_value_sharpe < 0.05:
                print("  ✅ PASS: Strategy timing is statistically significant (p < 0.05)")
            elif p_value_sharpe < 0.10:
                print("  ⚠️ MARGINAL: Some evidence of timing skill (p < 0.10)")
            else:
                print("  ❌ FAIL: No evidence strategy beats random entry timing")

        return results


# =============================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

class BootstrapAnalyzer:
    """
    Non-parametric bootstrap for Sharpe ratio confidence intervals.

    Method: Resample daily returns WITH replacement, compute Sharpe for
    each resample. Report percentile-based CI.
    """

    def __init__(self, n_resamples: int = 10000,
                 confidence_level: float = 0.95, random_seed: int = 42):
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_seed)

    def run(self, returns: pd.Series, verbose: bool = True) -> dict:
        """Bootstrap Sharpe ratio CI."""
        if verbose:
            print("\n" + "=" * 70)
            print("  3. BOOTSTRAP CONFIDENCE INTERVALS")
            print("=" * 70)

        ret = returns.dropna().values
        n = len(ret)
        if n < 30:
            if verbose:
                print("  ⚠️ Too few observations for bootstrap")
            return {'sharpe_ci_lower': 0, 'sharpe_ci_upper': 0, 'sharpe_point': 0}

        def calc_sharpe(r):
            if r.std() == 0:
                return 0
            ann = ((1 + r.mean()) ** 252 - 1)
            vol = r.std() * np.sqrt(252)
            return ann / vol if vol > 0 else 0

        point_sharpe = calc_sharpe(ret)
        boot_sharpes = np.zeros(self.n_resamples)

        for b in range(self.n_resamples):
            sample = self.rng.choice(ret, size=n, replace=True)
            boot_sharpes[b] = calc_sharpe(sample)

        alpha = 1 - self.confidence_level
        ci_lo = np.percentile(boot_sharpes, alpha / 2 * 100)
        ci_hi = np.percentile(boot_sharpes, (1 - alpha / 2) * 100)
        se = boot_sharpes.std()

        result = {
            'sharpe_point': point_sharpe,
            'sharpe_ci_lower': ci_lo,
            'sharpe_ci_upper': ci_hi,
            'sharpe_se': se,
            'n_resamples': self.n_resamples,
            'boot_sharpes': boot_sharpes,
        }

        if verbose:
            print(f"  Point Sharpe: {point_sharpe:.3f}")
            print(f"  Bootstrap SE: {se:.3f}")
            print(f"  {int(self.confidence_level*100)}% CI: "
                  f"[{ci_lo:.3f}, {ci_hi:.3f}]")
            if ci_lo > 0:
                print("  ✅ PASS: Sharpe ratio CI excludes zero")
            elif point_sharpe > 0:
                print("  ⚠️ MARGINAL: Point estimate positive but CI includes zero")
            else:
                print("  ❌ FAIL: Sharpe ratio not significantly positive")

        return result


# =============================================================================
# 4. STRESS TESTING
# =============================================================================

class StressTester:
    """Evaluate strategy performance during known crisis periods."""

    CRISIS_PERIODS = {
        '2018 Dec Selloff': ('2018-10-01', '2018-12-24'),
        '2020 COVID Crash': ('2020-02-19', '2020-03-23'),
        '2020 Recovery Vol': ('2020-03-24', '2020-06-30'),
        '2022 Rate Hikes': ('2022-01-01', '2022-06-30'),
        '2022 Bear Market': ('2022-06-01', '2022-10-12'),
        '2023 Banking Crisis': ('2023-03-01', '2023-04-15'),
    }

    def run(self, results: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print("\n" + "=" * 70)
            print("  4. STRESS TESTING (Crisis Period Analysis)")
            print("=" * 70)

        rows = []
        for name, (start, end) in self.CRISIS_PERIODS.items():
            mask = (results.index >= start) & (results.index <= end)
            crisis = results[mask]
            if len(crisis) < 5:
                continue

            s_ret = crisis['strategy_return'].dropna()
            m_ret = crisis['market_return'].dropna()
            s_total = (1 + s_ret).prod() - 1
            m_total = (1 + m_ret).prod() - 1
            trading_rate = (crisis['final_signal'] != 0).mean()
            regime_dist = crisis['regime'].value_counts(normalize=True)

            rows.append({
                'Crisis': name,
                'Start': start,
                'End': end,
                'Strategy_%': s_total * 100,
                'Market_%': m_total * 100,
                'Alpha_%': (s_total - m_total) * 100,
                'Trade_Rate_%': trading_rate * 100,
                'Crisis_Regime_%': regime_dist.get(0, 0) * 100,
                'Days': len(crisis),
            })

        df = pd.DataFrame(rows)
        if verbose and len(df) > 0:
            for _, row in df.iterrows():
                icon = "✅" if row['Strategy_%'] > row['Market_%'] else "⚠️"
                print(f"  {icon} {row['Crisis']}: Strategy={row['Strategy_%']:+.2f}% "
                      f"Market={row['Market_%']:+.2f}% "
                      f"Alpha={row['Alpha_%']:+.2f}%")

            avg_alpha = df['Alpha_%'].mean()
            if avg_alpha > 0:
                print(f"\n  ✅ PASS: Average crisis alpha = {avg_alpha:+.2f}%")
            elif df['Trade_Rate_%'].mean() < 30:
                print(f"\n  ✅ PASS: Strategy correctly avoids crises "
                      f"(avg trade rate {df['Trade_Rate_%'].mean():.0f}%)")
            else:
                print(f"\n  ❌ FAIL: Negative crisis alpha = {avg_alpha:.2f}%")

        return df


# =============================================================================
# 5. SENSITIVITY ANALYSIS
# =============================================================================

class SensitivityAnalyzer:
    """Grid search over parameter space to test robustness."""

    def run(self, stock_y: pd.Series, stock_x: pd.Series,
            market_index: pd.Series, train_end_date: str = '2020-12-31',
            entry_z_values: list = None, exit_z_values: list = None,
            z_windows: list = None,
            slippage_bps: float = 5.0, verbose: bool = True) -> pd.DataFrame:

        if entry_z_values is None:
            entry_z_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        if exit_z_values is None:
            exit_z_values = [0.2, 0.3, 0.5, 0.7, 1.0]
        if z_windows is None:
            z_windows = [30, 40, 60]

        if verbose:
            print("\n" + "=" * 70)
            print("  5. SENSITIVITY ANALYSIS (Parameter Robustness)")
            print("=" * 70)
            total = len(entry_z_values) * len(exit_z_values) * len(z_windows)
            print(f"  Grid: {len(entry_z_values)} entry × {len(exit_z_values)} exit "
                  f"× {len(z_windows)} window = {total} combos")

        rows = []
        done = 0

        for zw in z_windows:
            for ez in entry_z_values:
                for xz in exit_z_values:
                    if xz >= ez:
                        continue  # exit threshold must be < entry
                    done += 1
                    try:
                        sys = ConservativeSystem()
                        res = sys.run_backtest(
                            stock_y, stock_x, market_index,
                            train_end_date=train_end_date,
                            entry_z_normal=ez, exit_z_normal=xz,
                            entry_z_volatile=ez + 0.5,
                            exit_z_volatile=max(0.1, xz - 0.1),
                            z_score_window=zw,
                            slippage_bps=slippage_bps,
                            verbose=False
                        )
                        oos = res[res['period'] == 'oos']
                        ret = oos['strategy_return'].dropna()
                        tr = (1 + ret).prod() - 1
                        vol = ret.std() * np.sqrt(252)
                        ar = (1 + tr) ** (252 / max(len(ret), 1)) - 1
                        sh = ar / vol if vol > 0 else 0

                        trades = sys.get_trade_analysis(period='oos')
                        rows.append({
                            'entry_z': ez, 'exit_z': xz, 'z_window': zw,
                            'OOS_Return_%': tr * 100,
                            'OOS_Sharpe': sh,
                            'OOS_Trades': len(trades),
                            'OOS_WinRate_%': trades['profitable'].mean() * 100 if len(trades) > 0 else 0,
                        })
                    except Exception as e:
                        logger.warning(f"Sensitivity ({ez},{xz},{zw}): {e}")

        df = pd.DataFrame(rows)

        if verbose and len(df) > 0:
            pos_pct = (df['OOS_Return_%'] > 0).mean() * 100
            pos_sharpe_pct = (df['OOS_Sharpe'] > 0).mean() * 100
            sharpe_std = df['OOS_Sharpe'].std()
            print(f"\n  Combinations tested: {len(df)}")
            print(f"  Profitable combos (Return > 0): {pos_pct:.0f}%")
            print(f"  Positive Sharpe combos: {pos_sharpe_pct:.0f}%")
            print(f"  Sharpe range: [{df['OOS_Sharpe'].min():.2f}, {df['OOS_Sharpe'].max():.2f}]")
            print(f"  Sharpe mean: {df['OOS_Sharpe'].mean():.3f} ± {sharpe_std:.3f}")
            print(f"  Best combo: entry_z={df.loc[df['OOS_Sharpe'].idxmax(), 'entry_z']}, "
                  f"exit_z={df.loc[df['OOS_Sharpe'].idxmax(), 'exit_z']}, "
                  f"window={df.loc[df['OOS_Sharpe'].idxmax(), 'z_window']}")

            # Profitable region assessment
            if pos_pct >= 75:
                print(f"  ✅ ROBUST: {pos_pct:.0f}% profitable (>60% threshold)")
            elif pos_pct >= 60:
                print(f"  ⚠️ MARGINAL: {pos_pct:.0f}% profitable")
            elif pos_pct >= 30:
                print(f"  ⚠️ SENSITIVE: {pos_pct:.0f}% profitable")
            else:
                print(f"  ❌ FRAGILE: Only {pos_pct:.0f}% profitable (<30% threshold)")

            # Generate heatmap
            try:
                self._save_heatmap(df, z_windows)
                if verbose:
                    print(f"  📊 Heatmap saved: Research/results/sensitivity_heatmap.png")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ Heatmap failed: {e}")

        return df

    @staticmethod
    def _save_heatmap(df: pd.DataFrame, z_windows: list = None):
        """Generate Sharpe ratio heatmap across parameter space."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Research', 'results'
        )
        os.makedirs(results_dir, exist_ok=True)

        if z_windows is None:
            z_windows = sorted(df['z_window'].unique())

        n_windows = len(z_windows)
        fig, axes = plt.subplots(1, n_windows, figsize=(6 * n_windows, 5),
                                 squeeze=False)

        # Diverging colormap centered at zero
        vmin = df['OOS_Sharpe'].min()
        vmax = df['OOS_Sharpe'].max()
        abs_max = max(abs(vmin), abs(vmax), 0.01)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        for idx, zw in enumerate(z_windows):
            ax = axes[0, idx]
            sub = df[df['z_window'] == zw]

            if len(sub) == 0:
                ax.set_title(f'Window={zw} (no data)')
                continue

            pivot = sub.pivot_table(
                index='exit_z', columns='entry_z',
                values='OOS_Sharpe', aggfunc='first'
            )

            im = ax.imshow(
                pivot.values, aspect='auto', cmap='RdYlGn', norm=norm,
                origin='lower',
            )

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
            ax.set_xlabel('Entry Z', fontsize=10)
            ax.set_ylabel('Exit Z', fontsize=10)
            ax.set_title(f'Window = {zw}d', fontsize=11, fontweight='bold')

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > abs_max * 0.6 else 'black'
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                                fontsize=7, color=color, fontweight='bold')

        fig.colorbar(im, ax=axes[0, -1], shrink=0.8, label='OOS Sharpe')
        fig.suptitle('Sensitivity Analysis — OOS Sharpe by Parameter Combination',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(results_dir, 'sensitivity_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# 6. STATISTICAL TESTS
# =============================================================================

class StatisticalTester:
    """Standard statistical tests for return series."""

    def run(self, results: pd.DataFrame, period: str = 'oos',
            verbose: bool = True) -> dict:

        if period == 'oos':
            data = results[results['period'] == 'oos']
        elif period == 'train':
            data = results[results['period'] == 'train']
        else:
            data = results

        returns = data['strategy_return'].dropna()

        if verbose:
            print("\n" + "=" * 70)
            print(f"  6. STATISTICAL TESTS ({period.upper()} period)")
            print("=" * 70)

        test_results = {}

        # a) Jarque-Bera normality
        jb_stat, jb_pval = stats.jarque_bera(returns)
        test_results['jarque_bera'] = {'stat': jb_stat, 'pval': jb_pval}
        if verbose:
            print(f"  Jarque-Bera: stat={jb_stat:.2f}, p={jb_pval:.4f} "
                  f"({'Normal' if jb_pval > 0.05 else 'Non-normal (fat tails)'})")

        # b) One-sample t-test: mean > 0
        t_stat, t_pval = stats.ttest_1samp(returns, 0)
        one_sided_p = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2
        test_results['t_test'] = {'stat': t_stat, 'pval': one_sided_p}
        if verbose:
            sig = "✅ Significant" if one_sided_p < 0.05 else "❌ Not significant"
            print(f"  t-test (μ>0): t={t_stat:.3f}, p={one_sided_p:.4f} — {sig}")

        # c) Ljung-Box autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(returns, lags=[10], return_df=True)
            lb_pval = lb['lb_pvalue'].iloc[0]
            test_results['ljung_box'] = {'pval': lb_pval}
            if verbose:
                ac = "No autocorrelation" if lb_pval > 0.05 else "Autocorrelation detected"
                print(f"  Ljung-Box(10): p={lb_pval:.4f} — {ac}")
        except ImportError:
            test_results['ljung_box'] = {'pval': np.nan}

        # d) ADF on spread (should be stationary)
        try:
            from statsmodels.tsa.stattools import adfuller
            spread = data['spread'].dropna()
            adf_stat, adf_pval, *_ = adfuller(spread, maxlag=20)
            test_results['adf_spread'] = {'stat': adf_stat, 'pval': adf_pval}
            if verbose:
                st = "Stationary ✅" if adf_pval < 0.05 else "Non-stationary ❌"
                print(f"  ADF (spread): stat={adf_stat:.3f}, p={adf_pval:.4f} — {st}")
        except ImportError:
            test_results['adf_spread'] = {'pval': np.nan}

        # e) Skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()
        test_results['skewness'] = skew
        test_results['kurtosis'] = kurt
        if verbose:
            print(f"  Skewness: {skew:.3f} | Excess Kurtosis: {kurt:.3f}")

        return test_results


# =============================================================================
# 7. BENJAMINI-HOCHBERG MULTIPLE TESTING CORRECTION
# =============================================================================

def benjamini_hochberg(p_values: list, alpha: float = 0.05) -> list:
    """
    Benjamini-Hochberg procedure for controlling FDR.

    When testing multiple pairs, some will appear significant by chance.
    BH correction adjusts for this.

    Parameters
    ----------
    p_values : list of (pair_name, p_value) tuples
    alpha : float
        Desired FDR level.

    Returns
    -------
    list of (pair_name, p_value, adjusted_p, significant) tuples
    """
    sorted_pv = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_pv)
    results = []

    for i, (name, pv) in enumerate(sorted_pv, 1):
        bh_threshold = (i / m) * alpha
        adj_p = min(pv * m / i, 1.0)
        results.append((name, pv, adj_p, pv <= bh_threshold))

    return results


# =============================================================================
# 8. UNIFIED VALIDATOR
# =============================================================================

class StrategyValidator:
    """
    Unified validation suite. Runs all tests and produces a
    conference-grade validation report.
    """

    def __init__(self, system: ConservativeSystem,
                 ticker_y: str, ticker_x: str, config: dict = None):
        self.system = system
        self.ticker_y = ticker_y
        self.ticker_x = ticker_x
        self.config = config or load_config()
        self.validation_results = {}

    def run_all(self, stock_y: pd.Series, stock_x: pd.Series,
                market_index: pd.Series,
                train_end_date: str = '2020-12-31',
                verbose: bool = True) -> dict:
        """Run the complete validation suite."""

        cfg = self.config
        strat = cfg.get('strategy', {})
        mc_cfg = cfg.get('monte_carlo', {})
        wf_cfg = cfg.get('walk_forward', {})
        bs_cfg = cfg.get('bootstrap', {})
        sens_cfg = cfg.get('sensitivity', {})
        tc_cfg = cfg.get('transaction_costs', {})

        print("\n" + "=" * 70)
        print(f"  COMPREHENSIVE VALIDATION: {self.ticker_y}/{self.ticker_x}")
        print(f"  Train ≤ {train_end_date} | OOS > {train_end_date}")
        print("=" * 70)

        # 1. Walk-Forward
        wf = WalkForwardAnalyzer(
            min_train_days=wf_cfg.get('min_train_days', 504),
            step_days=wf_cfg.get('step_days', 63),
            test_days=wf_cfg.get('test_days', 126),
            min_folds=wf_cfg.get('min_folds', 8),
        )
        self.validation_results['walk_forward'] = wf.run(
            stock_y, stock_x, market_index,
            entry_z_normal=strat.get('entry_z_normal', 1.5),
            exit_z_normal=strat.get('exit_z_normal', 0.5),
            entry_z_volatile=strat.get('entry_z_volatile', 2.0),
            exit_z_volatile=strat.get('exit_z_volatile', 0.3),
            z_score_window=strat.get('z_score_window', 40),
            slippage_bps=tc_cfg.get('slippage_bps', 5.0),
            verbose=verbose,
        )

        # 2. Monte Carlo
        mc = MonteCarloValidator(
            n_simulations=mc_cfg.get('n_simulations', 10000),
            confidence_level=mc_cfg.get('confidence_level', 0.95),
            random_seed=mc_cfg.get('random_seed', 42),
        )
        self.validation_results['monte_carlo'] = mc.run(
            self.system, period='oos', verbose=verbose
        )

        # 3. Bootstrap
        oos_data = self.system.results[self.system.results['period'] == 'oos']
        bs = BootstrapAnalyzer(
            n_resamples=bs_cfg.get('n_resamples', 10000),
            confidence_level=bs_cfg.get('confidence_level', 0.95),
            random_seed=bs_cfg.get('random_seed', 42),
        )
        self.validation_results['bootstrap'] = bs.run(
            oos_data['strategy_return'], verbose=verbose
        )

        # 4. Stress Test
        st = StressTester()
        self.validation_results['stress_test'] = st.run(
            self.system.results, verbose=verbose
        )

        # 5. Sensitivity
        sa = SensitivityAnalyzer()
        self.validation_results['sensitivity'] = sa.run(
            stock_y, stock_x, market_index,
            train_end_date=train_end_date,
            entry_z_values=sens_cfg.get('entry_z_values', [1.0, 1.5, 2.0, 2.5]),
            exit_z_values=sens_cfg.get('exit_z_values', [0.3, 0.5, 0.7]),
            z_windows=sens_cfg.get('z_windows', [30, 40, 60]),
            slippage_bps=tc_cfg.get('slippage_bps', 5.0),
            verbose=verbose,
        )

        # 6. Statistical Tests
        self.validation_results['statistical'] = StatisticalTester().run(
            self.system.results, period='oos', verbose=verbose
        )

        return self.validation_results

    def compute_score(self) -> dict:
        """Compute validation scores for report."""
        scores = {}
        vr = self.validation_results

        # Walk-Forward
        wf = vr.get('walk_forward', pd.DataFrame())
        if len(wf) > 0:
            consistency = (wf['Return_%'] > 0).mean()
            avg_sharpe = wf['Sharpe'].mean()
            if consistency >= 0.6 and avg_sharpe > 0.3:
                scores['Walk-Forward'] = 100
            elif consistency >= 0.5 and avg_sharpe > 0:
                scores['Walk-Forward'] = 60
            else:
                scores['Walk-Forward'] = 20
        else:
            scores['Walk-Forward'] = 0

        # Monte Carlo
        mc = vr.get('monte_carlo', {})
        p = mc.get('p_value_sharpe', 1.0)
        if p < 0.01:
            scores['Monte Carlo'] = 100
        elif p < 0.05:
            scores['Monte Carlo'] = 80
        elif p < 0.10:
            scores['Monte Carlo'] = 60
        else:
            scores['Monte Carlo'] = 20

        # Bootstrap
        bs = vr.get('bootstrap', {})
        if bs.get('sharpe_ci_lower', 0) > 0:
            scores['Bootstrap CI'] = 100
        elif bs.get('sharpe_point', 0) > 0:
            scores['Bootstrap CI'] = 60
        else:
            scores['Bootstrap CI'] = 20

        # Stress Test
        st = vr.get('stress_test', pd.DataFrame())
        if len(st) > 0:
            avg_alpha = st['Alpha_%'].mean()
            if avg_alpha > 0:
                scores['Stress Test'] = 100
            elif st['Trade_Rate_%'].mean() < 30:
                scores['Stress Test'] = 80
            elif avg_alpha > -3:
                scores['Stress Test'] = 60
            else:
                scores['Stress Test'] = 20
        else:
            scores['Stress Test'] = 0

        # Sensitivity
        sens = vr.get('sensitivity', pd.DataFrame())
        if len(sens) > 0:
            pos_pct = (sens['OOS_Return_%'] > 0).mean()
            if pos_pct >= 0.75:
                scores['Sensitivity'] = 100
            elif pos_pct >= 0.60:
                scores['Sensitivity'] = 70
            else:
                scores['Sensitivity'] = 20
        else:
            scores['Sensitivity'] = 0

        # Statistical Tests
        stat = vr.get('statistical', {})
        t_pval = stat.get('t_test', {}).get('pval', 1.0)
        adf_pval = stat.get('adf_spread', {}).get('pval', 1.0)
        stat_score = 0
        if t_pval < 0.05:
            stat_score += 50
        elif t_pval < 0.10:
            stat_score += 25
        if adf_pval < 0.05:
            stat_score += 50
        elif adf_pval < 0.10:
            stat_score += 25
        scores['Statistical Tests'] = stat_score

        scores['Overall'] = np.mean(list(scores.values()))
        return scores

    def generate_report(self, save_path: str = None) -> str:
        """Generate a structured text validation report."""
        scores = self.compute_score()
        overall = scores.pop('Overall')

        lines = []
        lines.append("=" * 80)
        lines.append("PAIRS TRADING STRATEGY — VALIDATION REPORT")
        lines.append(f"Pair: {self.ticker_y} / {self.ticker_x}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        lines.append("VALIDATION SCORES")
        lines.append("-" * 80)
        for test, score in scores.items():
            icon = "✅" if score >= 80 else ("⚠️" if score >= 60 else "❌")
            lines.append(f"  {test:.<40} {score:>3.0f}/100  {icon}")
        lines.append("-" * 80)
        lines.append(f"  {'OVERALL':.<40} {overall:>3.0f}/100")
        lines.append("")

        # Detailed sections
        vr = self.validation_results

        # Walk-Forward detail
        wf = vr.get('walk_forward', pd.DataFrame())
        if len(wf) > 0:
            lines.append("WALK-FORWARD DETAIL")
            lines.append(f"  Folds: {len(wf)} | Avg Return: {wf['Return_%'].mean():.2f}% | "
                         f"Avg Sharpe: {wf['Sharpe'].mean():.3f} | "
                         f"Consistency: {(wf['Return_%'] > 0).mean()*100:.0f}%")
            lines.append("")

        # Monte Carlo detail
        mc = vr.get('monte_carlo', {})
        if mc.get('n_sims', 0) > 0:
            lines.append("MONTE CARLO DETAIL")
            lines.append(f"  Method: Random entry/exit dates (NOT shuffled returns)")
            lines.append(f"  Simulations: {mc['n_sims']} | "
                         f"Actual Sharpe: {mc['actual_sharpe']:.3f}")
            lines.append(f"  p-value (Sharpe): {mc['p_value_sharpe']:.4f} | "
                         f"p-value (Return): {mc['p_value_return']:.4f}")
            lines.append("")

        # Bootstrap detail
        bs = vr.get('bootstrap', {})
        if bs.get('n_resamples', 0) > 0:
            lines.append("BOOTSTRAP DETAIL")
            lines.append(f"  Sharpe: {bs['sharpe_point']:.3f} "
                         f"[{bs['sharpe_ci_lower']:.3f}, {bs['sharpe_ci_upper']:.3f}] "
                         f"(SE={bs['sharpe_se']:.3f})")
            lines.append("")

        # Recommendation
        lines.append("=" * 80)
        lines.append("RECOMMENDATION")
        lines.append("=" * 80)
        if overall >= 80:
            lines.append("✅ VALIDATED — proceed with paper trading")
        elif overall >= 60:
            lines.append("⚠️ MARGINAL — use with caution and smaller sizing")
        else:
            lines.append("❌ NOT VALIDATED — do not trade this pair")

        report = "\n".join(lines)
        print(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\n  Report saved: {save_path}")

        return report


# =============================================================================
# MAIN
# =============================================================================

def run_complete_validation(ticker_y: str = 'BAC', ticker_x: str = 'PNC',
                            start_date: str = '2015-01-01',
                            end_date: str = '2025-06-30',
                            train_end_date: str = '2020-12-31') -> tuple:
    """Run end-to-end validation for a single pair."""
    print(f"\n{'=' * 70}")
    print(f"  VALIDATING: {ticker_y} / {ticker_x}")
    print(f"{'=' * 70}")

    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    system = ConservativeSystem()
    system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        slippage_bps=5.0,
    )

    validator = StrategyValidator(system, ticker_y, ticker_x)
    validator.run_all(stock_y, stock_x, market,
                      train_end_date=train_end_date)

    report_name = f"Research/results/validation_{ticker_y}_{ticker_x}.txt"
    validator.generate_report(save_path=report_name)

    return validator, validator.compute_score()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    validator, scores = run_complete_validation('BAC', 'PNC')
    print(f"\nFinal Score: {scores.get('Overall', 0):.0f}/100")
