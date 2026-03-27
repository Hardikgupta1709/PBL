"""
Day-10 bootstrap block-size sensitivity.

Evaluates how Sharpe bootstrap CI width changes with block size
for top success pairs.

Outputs:
- Research/results/bootstrap_block_sensitivity.csv
- Paper/figures/fig25_bootstrap_block_sensitivity.pdf/png
- Paper/tables/table15_bootstrap_block_sensitivity.tex
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
from Research.dynamic_pair_selector import backtest_with_dynamic_filter
from Research.statistical_tests import MetricBootstrapper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "Research", "results")
FIGURES_DIR = os.path.join(ROOT, "Paper", "figures")
TABLES_DIR = os.path.join(ROOT, "Paper", "tables")

SUCCESS_CSV = os.path.join(RESULTS_DIR, "success_pairs_full.csv")

TRAIN_END = "2020-12-31"
START_DATE = "2015-01-01"
END_DATE = "2025-06-30"

BLOCK_SIZES = [3, 5, 10, 20]
N_RESAMPLES = 2000


def load_top_pairs(n_top=3):
    df = pd.read_csv(SUCCESS_CSV)
    return df.sort_values("F_Sharpe", ascending=False).head(n_top)["Pair"].tolist()


def run_pair_bootstrap(pair):
    ty, tx = pair.split("/")
    stock_y, stock_x, market = download_data(ty, tx, "SPY", START_DATE, END_DATE)
    result = backtest_with_dynamic_filter(
        stock_y, stock_x, market,
        train_end_date=TRAIN_END,
        slippage_bps=5.0,
        verbose=False,
    )
    oos = result["filtered"][result["filtered"]["period"] == "oos"]
    returns = oos["strategy_return"].dropna()
    return returns


def main():
    top_pairs = load_top_pairs()
    rows = []

    for pair in top_pairs:
        returns = run_pair_bootstrap(pair)
        for block_size in BLOCK_SIZES:
            bs = MetricBootstrapper(
                n_resamples=N_RESAMPLES,
                block_size=block_size,
                random_seed=42,
            )
            metrics = bs.bootstrap_all_metrics(returns, use_block=True, verbose=False)
            sharpe = metrics.get("Sharpe", {})
            if not sharpe:
                continue
            ci_low = sharpe["ci_lower"]
            ci_high = sharpe["ci_upper"]
            rows.append({
                "Pair": pair,
                "Block_Size": block_size,
                "Sharpe_Point": sharpe["point"],
                "Sharpe_CI_Low": ci_low,
                "Sharpe_CI_High": ci_high,
                "CI_Width": ci_high - ci_low,
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "bootstrap_block_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    save_table(df)
    save_figure(df, top_pairs)


def save_table(df):
    out_path = os.path.join(TABLES_DIR, "table15_bootstrap_block_sensitivity.tex")
    avg = df.groupby("Block_Size")["CI_Width"].median().reset_index()

    with open(out_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Bootstrap block-size sensitivity (median Sharpe CI width across top success pairs).}\n")
        f.write("\\label{tab:bootstrap_block_sensitivity}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{cc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Block size} & \\textbf{Median Sharpe CI width} \\\\\n+")
        f.write("\\hline\n")
        for _, row in avg.iterrows():
            f.write(f"{int(row['Block_Size'])} & {row['CI_Width']:.3f} \\\\\n+")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {out_path}")


def save_figure(df, top_pairs):
    fig, ax = plt.subplots(figsize=(7, 4))
    for pair in top_pairs:
        sub = df[df["Pair"] == pair]
        ax.plot(sub["Block_Size"], sub["CI_Width"], marker='o', label=pair)

    ax.set_xlabel('Block size (days)')
    ax.set_ylabel('Sharpe CI width')
    ax.set_title('Bootstrap Block-Size Sensitivity', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(fontsize=8)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig25_bootstrap_block_sensitivity.{ext}'))
    plt.close(fig)
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig25_bootstrap_block_sensitivity.pdf/png')}")


if __name__ == "__main__":
    main()
