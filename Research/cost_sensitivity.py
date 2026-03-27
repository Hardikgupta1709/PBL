"""
Day-10 cost sensitivity analysis.

Evaluates filtered Sharpe vs transaction cost for top success pairs.
Outputs:
- Research/results/cost_sensitivity.csv
- Paper/figures/fig23_cost_sensitivity.pdf/png
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import download_data
from Research.dynamic_pair_selector import backtest_with_dynamic_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "Research", "results")
FIGURES_DIR = os.path.join(ROOT, "Paper", "figures")

SUCCESS_CSV = os.path.join(RESULTS_DIR, "success_pairs_full.csv")

TRAIN_END = "2020-12-31"
START_DATE = "2015-01-01"
END_DATE = "2025-06-30"

COST_LEVELS = [0, 2, 5, 10, 20, 30]


def main() -> None:
    df = pd.read_csv(SUCCESS_CSV)
    top_pairs = df.sort_values("F_Sharpe", ascending=False).head(3)["Pair"].tolist()

    rows = []
    for pair in top_pairs:
        ty, tx = pair.split("/")
        stock_y, stock_x, market = download_data(ty, tx, "SPY", START_DATE, END_DATE)

        for cost in COST_LEVELS:
            result = backtest_with_dynamic_filter(
                stock_y, stock_x, market,
                train_end_date=TRAIN_END,
                slippage_bps=float(cost),
                verbose=False,
            )
            sharpe = result["metrics_filtered"]["oos"]["sharpe"]
            rows.append({
                "Pair": pair,
                "Cost_Bps": cost,
                "F_Sharpe": sharpe,
            })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "cost_sensitivity.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for pair in top_pairs:
        sub = out_df[out_df["Pair"] == pair]
        ax.plot(sub["Cost_Bps"], sub["F_Sharpe"], marker='o', label=pair)

    ax.axvline(5, color='#666', ls='--', lw=1, label='5 bps')
    ax.axvline(10, color='#999', ls='--', lw=1, label='10 bps')
    ax.set_xlabel('One-way transaction cost (bps)')
    ax.set_ylabel('Filtered OOS Sharpe')
    ax.set_title('Cost Sensitivity (Top Success Pairs)', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(fontsize=8)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'fig23_cost_sensitivity.{ext}'))
    plt.close(fig)
    print(f"Saved: {os.path.join(FIGURES_DIR, 'fig23_cost_sensitivity.pdf/png')}")


if __name__ == "__main__":
    main()
