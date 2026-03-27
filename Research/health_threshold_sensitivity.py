"""
Day-10 health-threshold sensitivity grid.

Evaluates median filtered Sharpe and MaxDD across success pairs for
combinations of ADF/Hurst/EG thresholds.

Outputs:
- Research/results/health_threshold_sensitivity.csv
- Paper/figures/fig24_health_sensitivity.pdf/png
- Paper/tables/table14_health_threshold_sensitivity.tex
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import download_data
from Research.dynamic_pair_selector import backtest_with_dynamic_filter, HealthThresholds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "Research", "results")
FIGURES_DIR = os.path.join(ROOT, "Paper", "figures")
TABLES_DIR = os.path.join(ROOT, "Paper", "tables")

SUCCESS_CSV = os.path.join(RESULTS_DIR, "success_pairs_selection.csv")

TRAIN_END = "2020-12-31"
START_DATE = "2015-01-01"
END_DATE = "2025-06-30"

ADF_GRID = [0.01, 0.05, 0.10]
HURST_GRID = [0.45, 0.50]
COINT_GRID = [0.05, 0.10]


def load_pairs():
    df = pd.read_csv(SUCCESS_CSV)
    return [p.replace('_', '/') for p in df["Pair"].tolist()]


def run_grid(pairs):
    rows = []
    for adf_p in ADF_GRID:
        for hurst_max in HURST_GRID:
            for coint_p in COINT_GRID:
                sharpes = []
                maxdds = []
                for pair in pairs:
                    ty, tx = pair.split('/')
                    stock_y, stock_x, market = download_data(ty, tx, "SPY", START_DATE, END_DATE)
                    thresholds = HealthThresholds(
                        adf_pvalue=adf_p,
                        hurst_max=hurst_max,
                        coint_pvalue=coint_p,
                    )
                    result = backtest_with_dynamic_filter(
                        stock_y, stock_x, market,
                        train_end_date=TRAIN_END,
                        thresholds=thresholds,
                        slippage_bps=5.0,
                        verbose=False,
                    )
                    mf = result["metrics_filtered"]["oos"]
                    sharpes.append(mf["sharpe"])
                    maxdds.append(mf["max_dd"])

                rows.append({
                    "ADF_p": adf_p,
                    "Hurst_max": hurst_max,
                    "Coint_p": coint_p,
                    "Median_Sharpe": float(np.median(sharpes)),
                    "Median_MaxDD": float(np.median(maxdds)),
                })

    return pd.DataFrame(rows)


def save_table(df):
    out_path = os.path.join(TABLES_DIR, "table14_health_threshold_sensitivity.tex")
    with open(out_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Health-threshold sensitivity (median filtered OOS metrics across success pairs).}\n")
        f.write("\\label{tab:health_threshold_sensitivity}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{ADF $p$} & \\textbf{Hurst max} & \\textbf{Coint $p$} & \\textbf{Median Sharpe} \\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            f.write(f"{row['ADF_p']:.2f} & {row['Hurst_max']:.2f} & {row['Coint_p']:.2f} & {row['Median_Sharpe']:+.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {out_path}")


def save_figure(df):
    fig, axes = plt.subplots(1, len(COINT_GRID), figsize=(10, 3.6), sharey=True)
    if len(COINT_GRID) == 1:
        axes = [axes]

    for idx, coint_p in enumerate(COINT_GRID):
        sub = df[df["Coint_p"] == coint_p]
        pivot = sub.pivot(index="ADF_p", columns="Hurst_max", values="Median_MaxDD")
        ax = axes[idx]
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(HURST_GRID)))
        ax.set_xticklabels([f"{h:.2f}" for h in HURST_GRID])
        ax.set_yticks(range(len(ADF_GRID)))
        ax.set_yticklabels([f"{a:.2f}" for a in ADF_GRID])
        ax.set_title(f"Coint p={coint_p:.2f}")
        ax.set_xlabel('Hurst max')
        if idx == 0:
            ax.set_ylabel('ADF p')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label='Median MaxDD')
    fig.suptitle('Health Threshold Sensitivity (Median MaxDD)', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig24_health_sensitivity.{ext}'))
    plt.close(fig)
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig24_health_sensitivity.pdf/png')}")


def main():
    pairs = load_pairs()
    df = run_grid(pairs)

    out_csv = os.path.join(RESULTS_DIR, "health_threshold_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    save_table(df)
    save_figure(df)


if __name__ == "__main__":
    main()
