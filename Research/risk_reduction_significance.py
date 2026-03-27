#!/usr/bin/env python3
"""
Risk Reduction Significance Tests (Expanded Universe)
====================================================
Aggregate, cross-sectional significance tests using per-pair OOS metrics.

Inputs:
  Research/results/expanded_universe_summary.csv

Outputs:
  Research/results/risk_reduction_significance.csv
  Research/results/risk_reduction_significance.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ks_2samp
from datetime import datetime
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Core_Strategy.conservative_strategy import download_data
from Research.dynamic_pair_selector import backtest_with_dynamic_filter
from Core_Strategy.strategy_validator import benjamini_hochberg

RESULTS_DIR = os.path.join(ROOT, "Research", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "expanded_universe_summary.csv")


def cvar(values: np.ndarray, alpha: float = 0.10) -> float:
    """Compute CVaR on the lower tail (alpha)."""
    if len(values) == 0:
        return np.nan
    threshold = np.quantile(values, alpha)
    tail = values[values <= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)


def main(pairwise: bool = False) -> None:
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    # Keep rows with filtered metrics
    df = df.dropna(subset=["UF_OOS_MaxDD_%", "F_OOS_MaxDD_%", "UF_OOS_Return_%", "F_OOS_Return_%"])
    if len(df) == 0:
        raise ValueError("No valid rows for risk tests.")

    # Max drawdown (use absolute magnitude)
    uf_dd = df["UF_OOS_MaxDD_%"].abs().values
    f_dd = df["F_OOS_MaxDD_%"].abs().values

    # Cross-sectional return distributions
    uf_ret = df["UF_OOS_Return_%"].values
    f_ret = df["F_OOS_Return_%"].values

    # Wilcoxon signed-rank on absolute drawdown reduction
    try:
        w_stat, w_p = wilcoxon(uf_dd, f_dd, alternative="greater")
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    # KS test on return distributions (filtered vs unfiltered)
    try:
        ks_stat, ks_p = ks_2samp(uf_ret, f_ret, alternative="two-sided")
    except ValueError:
        ks_stat, ks_p = np.nan, np.nan

    # CVaR (lower tail) on return distributions
    uf_cvar = cvar(uf_ret, alpha=0.10)
    f_cvar = cvar(f_ret, alpha=0.10)

    # Effect sizes
    median_dd_reduction = (np.median(uf_dd) - np.median(f_dd))
    median_ret_shift = (np.median(f_ret) - np.median(uf_ret))

    summary = {
        "n_pairs": len(df),
        "median_uf_maxdd_%": float(np.median(uf_dd)),
        "median_f_maxdd_%": float(np.median(f_dd)),
        "median_dd_reduction_%": float(median_dd_reduction),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_p_one_sided": float(w_p) if not np.isnan(w_p) else np.nan,
        "ks_stat": float(ks_stat) if not np.isnan(ks_stat) else np.nan,
        "ks_p_two_sided": float(ks_p) if not np.isnan(ks_p) else np.nan,
        "median_uf_return_%": float(np.median(uf_ret)),
        "median_f_return_%": float(np.median(f_ret)),
        "median_return_shift_%": float(median_ret_shift),
        "uf_cvar_10%": float(uf_cvar),
        "f_cvar_10%": float(f_cvar),
        "cvar_improvement_%": float(f_cvar - uf_cvar),
    }

    if pairwise:
        pvals = []
        for _, row in df.iterrows():
            pair = row["Pair"]
            try:
                ty, tx = pair.split("_")
            except ValueError:
                continue

            try:
                stock_y, stock_x, market = download_data(
                    ty, tx, "SPY", "2015-01-01", "2025-06-30"
                )
                res = backtest_with_dynamic_filter(
                    stock_y, stock_x, market,
                    train_end_date="2020-12-31",
                    slippage_bps=5.0,
                    apply_health_sizing=True,
                    sizing_thresholds=(0.40, 0.65, 0.80),
                    verbose=False,
                )
                uf_ret = res["unfiltered"].loc[
                    res["unfiltered"]["period"] == "oos", "strategy_return"
                ].dropna().values
                f_ret = res["filtered"].loc[
                    res["filtered"]["period"] == "oos", "strategy_return"
                ].dropna().values
                if len(uf_ret) < 30 or len(f_ret) < 30:
                    continue
                stat, p = ks_2samp(uf_ret, f_ret, alternative="two-sided")
                pvals.append((pair, p))
            except Exception:
                continue

        if len(pvals) > 0:
            bh = benjamini_hochberg(pvals, alpha=0.05)
            out_bh = os.path.join(RESULTS_DIR, "risk_reduction_pairwise_bh.csv")
            pd.DataFrame(
                [{"Pair": pair, "p_value": p, "p_bh": p_bh, "significant": sig}
                 for pair, p, p_bh, sig in bh]
            ).to_csv(out_bh, index=False)
            summary["pairwise_bh_significant"] = int(sum(1 for _, _, _, sig in bh if sig))
            summary["pairwise_bh_total"] = int(len(bh))

    # Save CSV summary
    out_csv = os.path.join(RESULTS_DIR, "risk_reduction_significance.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    # Save text report
    lines = []
    lines.append("=" * 72)
    lines.append("RISK REDUCTION SIGNIFICANCE (CROSS-SECTIONAL)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Pairs: {len(df)}")
    lines.append("=" * 72)
    lines.append("")

    lines.append("MAX DRAWDOWN (ABS)")
    lines.append(f"  Median UF MaxDD:  {summary['median_uf_maxdd_%']:.2f}%")
    lines.append(f"  Median F MaxDD:   {summary['median_f_maxdd_%']:.2f}%")
    lines.append(f"  Median Reduction: {summary['median_dd_reduction_%']:.2f}%")
    lines.append(f"  Wilcoxon (one-sided): stat={summary['wilcoxon_stat']:.3f}, p={summary['wilcoxon_p_one_sided']:.4f}")
    lines.append("")

    lines.append("RETURN DISTRIBUTION")
    lines.append(f"  Median UF Return: {summary['median_uf_return_%']:.2f}%")
    lines.append(f"  Median F Return:  {summary['median_f_return_%']:.2f}%")
    lines.append(f"  Median Shift:     {summary['median_return_shift_%']:.2f}%")
    lines.append(f"  KS test: stat={summary['ks_stat']:.3f}, p={summary['ks_p_two_sided']:.4f}")
    lines.append("")

    lines.append("TAIL RISK (CVaR 10%)")
    lines.append(f"  UF CVaR 10%: {summary['uf_cvar_10%']:.2f}%")
    lines.append(f"  F CVaR 10%:  {summary['f_cvar_10%']:.2f}%")
    lines.append(f"  CVaR Improvement: {summary['cvar_improvement_%']:.2f}%")

    out_txt = os.path.join(RESULTS_DIR, "risk_reduction_significance.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Risk reduction significance tests"
    )
    parser.add_argument("--pairwise", action="store_true",
                        help="Compute pairwise KS p-values with BH correction")
    args = parser.parse_args()
    main(pairwise=args.pairwise)
