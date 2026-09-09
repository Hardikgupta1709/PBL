import numpy as np
import pandas as pd
import os
import sys
import logging
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, download_data
)
from Research.baselines import run_all_baselines

logger = logging.getLogger(__name__)


# =============================================================================
# 1. BOOTSTRAP CI FOR ALL METRICS
# =============================================================================

class MetricBootstrapper:
    """
    Non-parametric bootstrap for all performance metrics.

    For each metric (Sharpe, total return, max drawdown, win rate),
    resamples daily returns with replacement B times and reports:
        - Point estimate
        - Standard error
        - Percentile-based 95% CI
        - Bias-corrected CI (BCa when feasible)

    For Sharpe ratio, uses the circular block bootstrap (block_size = 5)
    to preserve serial dependence, following Ledoit & Wolf (2008).
    """

    def __init__(self, n_resamples: int = 10000,
                 confidence_level: float = 0.95,
                 block_size: int = 5,
                 random_seed: int = 42):
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.block_size = block_size
        self.rng = np.random.RandomState(random_seed)

    # ----- Metric calculators -----

    @staticmethod
    def _sharpe(ret: np.ndarray) -> float:
        if len(ret) < 2 or ret.std() == 0:
            return 0.0
        ann_ret = (1 + ret.mean()) ** 252 - 1
        ann_vol = ret.std() * np.sqrt(252)
        return ann_ret / ann_vol if ann_vol > 0 else 0.0

    @staticmethod
    def _total_return(ret: np.ndarray) -> float:
        return float(np.prod(1 + ret) - 1)

    @staticmethod
    def _max_drawdown(ret: np.ndarray) -> float:
        cum = np.cumprod(1 + ret)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        return float(dd.min()) if len(dd) > 0 else 0.0

    @staticmethod
    def _win_rate(ret: np.ndarray) -> float:
        return float((ret > 0).mean()) if len(ret) > 0 else 0.0

    @staticmethod
    def _ann_return(ret: np.ndarray) -> float:
        if len(ret) < 2:
            return 0.0
        total = np.prod(1 + ret) - 1
        return float((1 + total) ** (252 / len(ret)) - 1) if total > -1 else -1.0

    # ----- Block bootstrap resampler -----

    def _block_resample(self, data: np.ndarray) -> np.ndarray:
        """Circular block bootstrap — preserves serial correlation."""
        n = len(data)
        n_blocks = int(np.ceil(n / self.block_size))
        starts = self.rng.randint(0, n, size=n_blocks)
        blocks = [data[s:s + self.block_size] if s + self.block_size <= n
                  else np.concatenate([data[s:], data[:s + self.block_size - n]])
                  for s in starts]
        return np.concatenate(blocks)[:n]

    def _iid_resample(self, data: np.ndarray) -> np.ndarray:
        return self.rng.choice(data, size=len(data), replace=True)

    # ----- Main bootstrap -----

    def bootstrap_all_metrics(self, returns: pd.Series,
                              use_block: bool = True,
                              verbose: bool = True) -> Dict:
        """
        Bootstrap all performance metrics from a return series.

        Parameters
        ----------
        returns : pd.Series
            Daily strategy returns.
        use_block : bool
            If True, uses circular block bootstrap for Sharpe (preserves
            autocorrelation). IID bootstrap for other metrics.
        """
        ret = returns.dropna().values
        n = len(ret)

        if n < 30:
            if verbose:
                print("  ⚠️ Too few observations for reliable bootstrap")
            return {}

        metrics = {
            'Sharpe': self._sharpe,
            'Total_Return': self._total_return,
            'Ann_Return': self._ann_return,
            'Max_Drawdown': self._max_drawdown,
            'Win_Rate': self._win_rate,
        }

        alpha = 1 - self.confidence_level
        results = {}

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  BOOTSTRAP CONFIDENCE INTERVALS ({self.n_resamples} resamples)")
            print(f"{'=' * 70}")

        for name, func in metrics.items():
            point = func(ret)

            # Choose resampler
            resampler = self._block_resample if (use_block and name == 'Sharpe') \
                else self._iid_resample

            boot_vals = np.zeros(self.n_resamples)
            for b in range(self.n_resamples):
                sample = resampler(ret)
                boot_vals[b] = func(sample)

            ci_lo = np.percentile(boot_vals, alpha / 2 * 100)
            ci_hi = np.percentile(boot_vals, (1 - alpha / 2) * 100)
            se = boot_vals.std()
            bias = boot_vals.mean() - point

            results[name] = {
                'point': point,
                'se': se,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                'bias': bias,
                'boot_values': boot_vals,
            }

            if verbose:
                # Format display
                if 'Return' in name:
                    print(f"  {name:.<25} {point*100:>8.2f}%  "
                          f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]  SE={se*100:.2f}%")
                elif name == 'Max_Drawdown':
                    print(f"  {name:.<25} {point*100:>8.2f}%  "
                          f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]  SE={se*100:.2f}%")
                elif name == 'Win_Rate':
                    print(f"  {name:.<25} {point*100:>8.1f}%   "
                          f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]  SE={se*100:.1f}%")
                else:
                    print(f"  {name:.<25} {point:>8.3f}   "
                          f"[{ci_lo:.3f}, {ci_hi:.3f}]  SE={se:.3f}")

        return results


# =============================================================================
# 2. PAIRED BOOTSTRAP TEST (Our Method vs Baseline)
# =============================================================================

class PairedBootstrapTest:
    """
    Paired bootstrap test for Sharpe ratio differences.

    H₀: Sharpe(ours) − Sharpe(baseline) = 0
    H₁: Sharpe(ours) − Sharpe(baseline) ≠ 0  (two-sided)

    Uses the SAME resampled indices for both return series to preserve
    correlation structure (paired test).

    Also computes Cohen's d effect size for practical significance.

    Reference:
        Ledoit & Wolf (2008) "Robust Performance Hypothesis Testing
        with the Sharpe Ratio"
    """

    def __init__(self, n_resamples: int = 10000, random_seed: int = 42):
        self.n_resamples = n_resamples
        self.rng = np.random.RandomState(random_seed)

    @staticmethod
    def _sharpe(ret: np.ndarray) -> float:
        if len(ret) < 2 or ret.std() == 0:
            return 0.0
        ann_ret = (1 + ret.mean()) ** 252 - 1
        ann_vol = ret.std() * np.sqrt(252)
        return ann_ret / ann_vol if ann_vol > 0 else 0.0

    def test(self, returns_ours: np.ndarray, returns_baseline: np.ndarray,
             baseline_name: str = 'Baseline',
             verbose: bool = True) -> Dict:
        """
        Test whether our Sharpe differs from baseline Sharpe.

        Both return arrays must have the same length (same dates).
        """
        n = min(len(returns_ours), len(returns_baseline))
        ret_a = returns_ours[:n]
        ret_b = returns_baseline[:n]

        sharpe_a = self._sharpe(ret_a)
        sharpe_b = self._sharpe(ret_b)
        observed_diff = sharpe_a - sharpe_b

        # Paired bootstrap: same resampled indices for both
        boot_diffs = np.zeros(self.n_resamples)
        for i in range(self.n_resamples):
            idx = self.rng.randint(0, n, size=n)
            boot_a = self._sharpe(ret_a[idx])
            boot_b = self._sharpe(ret_b[idx])
            boot_diffs[i] = boot_a - boot_b

        # Two-sided p-value
        # Under H0: center the distribution at 0
        centered = boot_diffs - boot_diffs.mean()
        p_value = (np.abs(centered) >= np.abs(observed_diff)).mean()

        ci_lo = np.percentile(boot_diffs, 2.5)
        ci_hi = np.percentile(boot_diffs, 97.5)
        se = boot_diffs.std()

        # Cohen's d effect size
        # d = mean_diff / pooled_sd
        pooled_sd = se if se > 0 else 1e-8
        cohens_d = observed_diff / pooled_sd

        result = {
            'baseline': baseline_name,
            'sharpe_ours': sharpe_a,
            'sharpe_baseline': sharpe_b,
            'sharpe_diff': observed_diff,
            'diff_ci_lower': ci_lo,
            'diff_ci_upper': ci_hi,
            'diff_se': se,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant_05': p_value < 0.05,
            'boot_diffs': boot_diffs,
        }

        if verbose:
            sig = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
            eff = _interpret_cohens_d(cohens_d)
            print(f"  vs {baseline_name:.<30} "
                  f"Δ={observed_diff:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
                  f"p={p_value:.4f} {sig}  d={cohens_d:.2f} ({eff})")

        return result


def _interpret_cohens_d(d: float) -> str:
    """Cohen (1988) effect size interpretation."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# 3. MULTIPLE TESTING CORRECTION
# =============================================================================

class MultipleTestingCorrector:
    """
    Correct for multiple comparisons across pairs and baselines.

    When testing 50 pairs × 6 baselines = 300 tests, some will be
    significant by chance. This class applies:

        1. Benjamini-Hochberg (BH) — controls False Discovery Rate (FDR)
        2. Bonferroni — controls Family-Wise Error Rate (FWER, conservative)
        3. Holm-Bonferroni — step-down version (less conservative)

    References:
        Benjamini & Hochberg (1995)
        Harvey, Liu & Zhu (2016)
    """

    @staticmethod
    def benjamini_hochberg(p_values: List[Tuple[str, float]],
                           alpha: float = 0.05) -> pd.DataFrame:
        """
        Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        p_values : list of (label, p_value) tuples
        alpha : float
            Desired FDR level.

        Returns
        -------
        DataFrame with columns: Label, Raw_p, Adjusted_p, BH_Significant
        """
        sorted_pv = sorted(p_values, key=lambda x: x[1])
        m = len(sorted_pv)
        rows = []

        # Step-up procedure
        max_significant_rank = 0
        for i, (label, pv) in enumerate(sorted_pv, 1):
            threshold = (i / m) * alpha
            if pv <= threshold:
                max_significant_rank = i

        for i, (label, pv) in enumerate(sorted_pv, 1):
            adj_p = min(pv * m / i, 1.0)
            rows.append({
                'Label': label,
                'Raw_p': pv,
                'BH_Threshold': (i / m) * alpha,
                'Adjusted_p': adj_p,
                'BH_Significant': i <= max_significant_rank,
                'Rank': i,
            })

        # Fix adjusted p-values to be monotone
        df = pd.DataFrame(rows)
        adj_p_corrected = df['Adjusted_p'].values.copy()
        for i in range(len(adj_p_corrected) - 2, -1, -1):
            adj_p_corrected[i] = min(adj_p_corrected[i], adj_p_corrected[i + 1])
        df['Adjusted_p'] = np.minimum(adj_p_corrected, 1.0)

        return df

    @staticmethod
    def bonferroni(p_values: List[Tuple[str, float]],
                   alpha: float = 0.05) -> pd.DataFrame:
        """
        Bonferroni correction — most conservative.
        Adjusted p = raw_p × m.
        """
        m = len(p_values)
        rows = []
        for label, pv in p_values:
            adj_p = min(pv * m, 1.0)
            rows.append({
                'Label': label,
                'Raw_p': pv,
                'Bonf_Adjusted_p': adj_p,
                'Bonf_Significant': adj_p < alpha,
            })
        return pd.DataFrame(rows).sort_values('Raw_p')

    @staticmethod
    def holm_bonferroni(p_values: List[Tuple[str, float]],
                        alpha: float = 0.05) -> pd.DataFrame:
        """
        Holm-Bonferroni step-down procedure — less conservative than Bonferroni.
        """
        sorted_pv = sorted(p_values, key=lambda x: x[1])
        m = len(sorted_pv)
        rows = []
        rejected_so_far = True

        for i, (label, pv) in enumerate(sorted_pv):
            adj_p = min(pv * (m - i), 1.0)
            significant = rejected_so_far and (pv < alpha / (m - i))
            if not significant:
                rejected_so_far = False
            rows.append({
                'Label': label,
                'Raw_p': pv,
                'Holm_Adjusted_p': adj_p,
                'Holm_Significant': significant,
            })

        # Enforce monotonicity
        df = pd.DataFrame(rows)
        adj = df['Holm_Adjusted_p'].values.copy()
        for i in range(1, len(adj)):
            adj[i] = max(adj[i], adj[i - 1])
        df['Holm_Adjusted_p'] = np.minimum(adj, 1.0)

        return df


# =============================================================================
# 4. FULL PIPELINE: BOOTSTRAP + BASELINES + MTC
# =============================================================================

def run_full_statistical_analysis(
    ticker_y: str = 'BAC', ticker_x: str = 'PNC',
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    n_bootstrap: int = 10000,
    verbose: bool = True
) -> Dict:
    """
    Run the complete statistical testing suite for one pair:

    1. Bootstrap CI for our strategy metrics
    2. Run all baselines
    3. Paired bootstrap tests (ours vs each baseline)
    4. Collect all p-values for downstream MTC
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  STATISTICAL ANALYSIS: {ticker_y}/{ticker_x}")
        print(f"{'=' * 70}")

    # --- Run our strategy ---
    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )

    system = ConservativeSystem()
    data = system.run_backtest(
        stock_y, stock_x, market,
        train_end_date=train_end_date,
        slippage_bps=5.0,
        verbose=False,
    )

    oos = data[data['period'] == 'oos']
    our_returns = oos['strategy_return'].dropna()

    results = {}

    # --- 1. Bootstrap CIs for our metrics ---
    bootstrapper = MetricBootstrapper(
        n_resamples=n_bootstrap, random_seed=42
    )
    results['bootstrap_ci'] = bootstrapper.bootstrap_all_metrics(
        our_returns, verbose=verbose
    )

    # --- 2. Run baselines ---
    comparison_df, baseline_results = run_all_baselines(
        ticker_y, ticker_x, start_date=start_date, end_date=end_date,
        train_end_date=train_end_date, verbose=False
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  PAIRED BOOTSTRAP TESTS (Ours vs Each Baseline)")
        print(f"{'=' * 70}")

    # --- 3. Paired bootstrap tests ---
    tester = PairedBootstrapTest(n_resamples=n_bootstrap, random_seed=42)
    paired_results = []

    our_ret_arr = our_returns.values

    for bname, bdata in baseline_results.items():
        if bname == 'full_system':
            continue

        bret = bdata.get('oos_returns', None)
        if bret is None or len(bret) == 0:
            continue

        bret_arr = bret.values if hasattr(bret, 'values') else np.array(bret)

        # Align lengths
        min_len = min(len(our_ret_arr), len(bret_arr))
        if min_len < 30:
            continue

        pair_result = tester.test(
            our_ret_arr[:min_len], bret_arr[:min_len],
            baseline_name=bname, verbose=verbose,
        )
        paired_results.append(pair_result)

    results['paired_tests'] = paired_results
    results['comparison'] = comparison_df

    # --- 4. Collect p-values ---
    p_values = [
        (f"{ticker_y}_{ticker_x}_vs_{r['baseline']}", r['p_value'])
        for r in paired_results
    ]
    results['p_values'] = p_values

    return results


def run_statistical_analysis_multi_pair(
    pairs: List[Tuple[str, str]],
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    n_bootstrap: int = 5000,
    fdr_alpha: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Run statistical analysis across multiple pairs and apply MTC.

    This is the main multi-pair entry point that produces:
    1. Per-pair bootstrap CIs
    2. Per-pair paired bootstrap tests vs baselines
    3. BH + Bonferroni correction across ALL tests
    4. Summary: "X of Y tests remain significant at FDR = 5%"
    """
    all_p_values = []
    all_bootstrap_cis = []
    all_paired_tests = []

    for i, (ty, tx) in enumerate(pairs):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(pairs)}] {ty}/{tx}")
        try:
            result = run_full_statistical_analysis(
                ty, tx, start_date, end_date, train_end_date,
                n_bootstrap=n_bootstrap, verbose=False,
            )

            # Collect bootstrap CIs
            if 'bootstrap_ci' in result and result['bootstrap_ci']:
                ci = result['bootstrap_ci']
                row = {'Pair': f"{ty}_{tx}"}
                for metric, vals in ci.items():
                    row[f'{metric}_point'] = vals['point']
                    row[f'{metric}_ci_lo'] = vals['ci_lower']
                    row[f'{metric}_ci_hi'] = vals['ci_upper']
                all_bootstrap_cis.append(row)

                sh = ci.get('Sharpe', {})
                print(f"    Sharpe: {sh.get('point', 0):.3f} "
                      f"[{sh.get('ci_lower', 0):.3f}, {sh.get('ci_upper', 0):.3f}]")

            # Collect paired tests
            for pt in result.get('paired_tests', []):
                pt_row = {
                    'Pair': f"{ty}_{tx}",
                    'Baseline': pt['baseline'],
                    'Sharpe_Ours': pt['sharpe_ours'],
                    'Sharpe_Baseline': pt['sharpe_baseline'],
                    'Diff': pt['sharpe_diff'],
                    'p_value': pt['p_value'],
                    'Cohens_d': pt['cohens_d'],
                }
                all_paired_tests.append(pt_row)

            # Collect p-values for MTC
            all_p_values.extend(result.get('p_values', []))

        except Exception as e:
            print(f"    ✗ Error: {e}")

    # --- Apply Multiple Testing Corrections ---
    print(f"\n{'=' * 70}")
    print(f"  MULTIPLE TESTING CORRECTION ({len(all_p_values)} tests)")
    print(f"{'=' * 70}")

    corrector = MultipleTestingCorrector()

    bh_results = pd.DataFrame()
    bonf_results = pd.DataFrame()

    if len(all_p_values) > 0:
        bh_results = corrector.benjamini_hochberg(all_p_values, alpha=fdr_alpha)
        bonf_results = corrector.bonferroni(all_p_values, alpha=fdr_alpha)
        holm_results = corrector.holm_bonferroni(all_p_values, alpha=fdr_alpha)

        # Summary
        n_raw_sig = sum(1 for _, p in all_p_values if p < fdr_alpha)
        n_bh_sig = int(bh_results['BH_Significant'].sum())
        n_bonf_sig = int(bonf_results['Bonf_Significant'].sum())
        n_holm_sig = int(holm_results['Holm_Significant'].sum())
        n_total = len(all_p_values)

        print(f"\n  Total tests: {n_total}")
        print(f"  Raw significant (p < {fdr_alpha}):        {n_raw_sig}/{n_total} "
              f"({n_raw_sig/n_total*100:.1f}%)")
        print(f"  BH-corrected (FDR = {fdr_alpha}):          {n_bh_sig}/{n_total} "
              f"({n_bh_sig/n_total*100:.1f}%)")
        print(f"  Holm-Bonferroni:                   {n_holm_sig}/{n_total} "
              f"({n_holm_sig/n_total*100:.1f}%)")
        print(f"  Bonferroni (most conservative):     {n_bonf_sig}/{n_total} "
              f"({n_bonf_sig/n_total*100:.1f}%)")

        if n_bh_sig > 0:
            print(f"\n  ✅ {n_bh_sig} test(s) remain significant after BH correction")
        else:
            print(f"\n  ⚠️ No tests survive BH correction — consider larger universe")

    # Build output
    ci_df = pd.DataFrame(all_bootstrap_cis) if all_bootstrap_cis else pd.DataFrame()
    pt_df = pd.DataFrame(all_paired_tests) if all_paired_tests else pd.DataFrame()

    return {
        'bootstrap_cis': ci_df,
        'paired_tests': pt_df,
        'bh_results': bh_results,
        'bonf_results': bonf_results,
        'all_p_values': all_p_values,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Single pair demo
    results = run_full_statistical_analysis('BAC', 'PNC')

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save bootstrap CIs
    if results.get('bootstrap_ci'):
        rows = []
        for metric, vals in results['bootstrap_ci'].items():
            rows.append({
                'Metric': metric,
                'Point': vals['point'],
                'CI_Lower': vals['ci_lower'],
                'CI_Upper': vals['ci_upper'],
                'SE': vals['se'],
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(results_dir, 'bootstrap_ci_BAC_PNC.csv'), index=False
        )

    # Save paired test results
    if results.get('paired_tests'):
        rows = [{
            'Baseline': r['baseline'],
            'Sharpe_Ours': r['sharpe_ours'],
            'Sharpe_Baseline': r['sharpe_baseline'],
            'Diff': r['sharpe_diff'],
            'CI_Lower': r['diff_ci_lower'],
            'CI_Upper': r['diff_ci_upper'],
            'p_value': r['p_value'],
            'Cohens_d': r['cohens_d'],
            'Significant': r['significant_05'],
        } for r in results['paired_tests']]
        pd.DataFrame(rows).to_csv(
            os.path.join(results_dir, 'paired_tests_BAC_PNC.csv'), index=False
        )

    print("\n  Results saved to Research/results/")
