"""
Day-7 operational feasibility metrics (full scale).

Computes trading activity metrics for filtered signals across the 60-pair universe:
- Trades per year
- Average holding period (days)
- In-position fraction

Outputs:
- Research/results/operational_metrics_pairs.csv
- Research/results/operational_metrics_summary.csv
- Paper/figures/fig22_operational_metrics.pdf/png
- Paper/tables/table12_operational_metrics.tex
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'Research', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'Paper', 'figures')
TABLES_DIR = os.path.join(BASE_DIR, 'Paper', 'tables')

PAIR_SUMMARY_CSV = os.path.join(RESULTS_DIR, 'expanded_universe_summary.csv')

TRAIN_END = '2020-12-31'
START_DATE = '2015-01-01'
END_DATE = '2025-06-30'


def load_pairs():
    df = pd.read_csv(PAIR_SUMMARY_CSV)
    pairs = df['Pair'].dropna().unique().tolist()
    out = []
    for pair in pairs:
        if '_' in pair:
            ty, tx = pair.split('_')
            out.append((ty, tx))
    return out


def compute_activity_metrics(signal: pd.Series, n_days: int) -> dict:
    sig = signal.fillna(0)
    in_pos = (sig.abs() > 0)
    in_pos_frac = float(in_pos.mean()) if n_days else 0.0

    # Trade count: count transitions from 0 to non-zero
    changes = sig.diff().fillna(0)
    entries = (changes != 0) & (sig != 0)
    n_trades = int(entries.sum())

    # Holding periods: lengths of non-zero runs
    holds = []
    current = 0
    for val in sig.values:
        if val != 0:
            current += 1
        elif current > 0:
            holds.append(current)
            current = 0
    if current > 0:
        holds.append(current)

    avg_hold = float(np.mean(holds)) if holds else 0.0
    med_hold = float(np.median(holds)) if holds else 0.0

    years = n_days / 252 if n_days else 0.0
    trades_per_year = float(n_trades / years) if years else 0.0

    return {
        'Trades': n_trades,
        'Trades_Per_Year': trades_per_year,
        'Avg_Hold_Days': avg_hold,
        'Med_Hold_Days': med_hold,
        'In_Position_Frac': in_pos_frac,
    }


def save_table(summary: dict):
    table_path = os.path.join(TABLES_DIR, 'table12_operational_metrics.tex')

    def fmt(x):
        return f"{x:.2f}"

    with open(table_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Operational feasibility metrics across 60 pairs (filtered OOS).}\n")
        f.write("\\label{tab:operational_metrics}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Median} & \\textbf{P25} & \\textbf{P75} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Trades per year & {fmt(summary['Trades_Per_Year_Med'])} & {fmt(summary['Trades_Per_Year_P25'])} & {fmt(summary['Trades_Per_Year_P75'])} \\\\\n")
        f.write(f"Avg holding (days) & {fmt(summary['Avg_Hold_Days_Med'])} & {fmt(summary['Avg_Hold_Days_P25'])} & {fmt(summary['Avg_Hold_Days_P75'])} \\\\\n")
        f.write(f"In-position fraction & {fmt(summary['In_Position_Frac_Med'])} & {fmt(summary['In_Position_Frac_P25'])} & {fmt(summary['In_Position_Frac_P75'])} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  ✓ Saved {table_path}")


def save_figure(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    axes[0].boxplot(df['Trades_Per_Year'].values, labels=['Trades/yr'])
    axes[0].set_title('Trade Frequency')
    axes[0].set_ylabel('Trades per year')

    axes[1].boxplot(df['Avg_Hold_Days'].values, labels=['Hold'])
    axes[1].set_title('Holding Period')
    axes[1].set_ylabel('Days')

    axes[2].boxplot(df['In_Position_Frac'].values, labels=['In-position'])
    axes[2].set_title('Market Exposure')
    axes[2].set_ylabel('Fraction')

    for ax in axes:
        ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = os.path.join(FIGURES_DIR, f'fig22_operational_metrics.{ext}')
        fig.savefig(out_path)
    plt.close(fig)

    print(f"  ✓ Saved {os.path.join(FIGURES_DIR, 'fig22_operational_metrics.pdf/png')}")


def main():
    print("\n" + "=" * 70)
    print("  DAY 7: OPERATIONAL METRICS (FULL SCALE)")
    print("=" * 70)

    pairs = load_pairs()
    if not pairs:
        raise RuntimeError("No pairs found in expanded_universe_summary.csv")

    rows = []
    skipped = []
    for i, (ty, tx) in enumerate(pairs, 1):
        print(f"  [{i:02d}/{len(pairs)}] {ty}/{tx}")
        try:
            stock_y, stock_x, market = download_data(ty, tx, 'SPY', START_DATE, END_DATE)
        except Exception as e:
            skipped.append(f"{ty}/{tx}")
            print(f"    ⚠ Skipping {ty}/{tx}: {e}")
            continue

        if stock_y is None or stock_x is None or market is None:
            skipped.append(f"{ty}/{tx}")
            print(f"    ⚠ Skipping {ty}/{tx}: missing data")
            continue
        result = backtest_with_dynamic_filter(
            stock_y, stock_x, market,
            train_end_date=TRAIN_END,
            health_step=5,
            verbose=False,
            slippage_bps=5.0,
        )

        filtered_bt = result['filtered']
        oos = filtered_bt[filtered_bt['period'] == 'oos']
        signal = oos['final_signal']
        n_days = len(oos)

        metrics = compute_activity_metrics(signal, n_days)
        rows.append({
            'Pair': f"{ty}/{tx}",
            **metrics,
        })

    df = pd.DataFrame(rows)
    if skipped:
        print(f"  Skipped pairs: {len(skipped)}")
    pair_out = os.path.join(RESULTS_DIR, 'operational_metrics_pairs.csv')
    df.to_csv(pair_out, index=False)
    print(f"\n  ✓ Saved {pair_out}")

    summary = {
        'Trades_Per_Year_Med': float(df['Trades_Per_Year'].median()),
        'Trades_Per_Year_P25': float(df['Trades_Per_Year'].quantile(0.25)),
        'Trades_Per_Year_P75': float(df['Trades_Per_Year'].quantile(0.75)),
        'Avg_Hold_Days_Med': float(df['Avg_Hold_Days'].median()),
        'Avg_Hold_Days_P25': float(df['Avg_Hold_Days'].quantile(0.25)),
        'Avg_Hold_Days_P75': float(df['Avg_Hold_Days'].quantile(0.75)),
        'In_Position_Frac_Med': float(df['In_Position_Frac'].median()),
        'In_Position_Frac_P25': float(df['In_Position_Frac'].quantile(0.25)),
        'In_Position_Frac_P75': float(df['In_Position_Frac'].quantile(0.75)),
    }

    summary_df = pd.DataFrame([summary])
    summary_out = os.path.join(RESULTS_DIR, 'operational_metrics_summary.csv')
    summary_df.to_csv(summary_out, index=False)
    print(f"  ✓ Saved {summary_out}")

    save_table(summary)
    save_figure(df)


if __name__ == '__main__':
    main()
