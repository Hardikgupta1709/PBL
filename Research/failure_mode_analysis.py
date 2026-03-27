"""
Full-scale failure mode analysis (Day 6).

Computes pre- vs post-2020 changes in:
- Rolling cointegration fraction
- Rolling return correlation
- Rolling Hurst exponent (spread)

Outputs:
- Research/results/failure_mode_pair_stats.csv
- Research/results/failure_mode_summary.csv
- Paper/figures/fig21_failure_modes.pdf/png
- Paper/tables/table11_failure_modes.tex
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
from Research.cointegration_analysis import rolling_cointegration, rolling_hurst

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'Research', 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'Paper', 'figures')
TABLES_DIR = os.path.join(BASE_DIR, 'Paper', 'tables')

PAIR_SUMMARY_CSV = os.path.join(RESULTS_DIR, 'expanded_universe_summary.csv')

TRAIN_END = pd.Timestamp('2020-12-31')
START_DATE = '2015-01-01'
END_DATE = '2025-06-30'

ROLL_WINDOW = 126
ROLL_STEP = 5


def load_pairs():
    df = pd.read_csv(PAIR_SUMMARY_CSV)
    pairs = df['Pair'].dropna().unique().tolist()
    # Pairs are formatted like "AAA_BBB"
    out = []
    for pair in pairs:
        if '_' in pair:
            ty, tx = pair.split('_')
            out.append((ty, tx))
    return out


def rolling_corr(returns_y, returns_x, window=126):
    return returns_y.rolling(window).corr(returns_x)


def summarize_pair(ty, tx):
    stock_y, stock_x, _ = download_data(ty, tx, 'SPY', START_DATE, END_DATE)
    df = pd.DataFrame({'Y': stock_y, 'X': stock_x}).dropna()

    # Returns for correlation
    ret_y = np.log(df['Y']).diff()
    ret_x = np.log(df['X']).diff()
    corr = rolling_corr(ret_y, ret_x, window=ROLL_WINDOW).dropna()

    # Rolling cointegration and Hurst on spread
    coint_df = rolling_cointegration(df['Y'], df['X'], window=ROLL_WINDOW, step=ROLL_STEP)
    spread = df['Y'] - df['X']
    hurst_df = rolling_hurst(spread, window=ROLL_WINDOW, step=ROLL_STEP)

    # Align to dates
    corr_pre = corr[corr.index <= TRAIN_END]
    corr_post = corr[corr.index > TRAIN_END]

    coint_pre = coint_df[coint_df.index <= TRAIN_END]
    coint_post = coint_df[coint_df.index > TRAIN_END]

    hurst_pre = hurst_df[hurst_df.index <= TRAIN_END]
    hurst_post = hurst_df[hurst_df.index > TRAIN_END]

    # Fractions and medians
    coint_frac_pre = float(coint_pre['is_cointegrated'].mean()) if len(coint_pre) else 0.0
    coint_frac_post = float(coint_post['is_cointegrated'].mean()) if len(coint_post) else 0.0

    corr_mean_pre = float(corr_pre.mean()) if len(corr_pre) else 0.0
    corr_mean_post = float(corr_post.mean()) if len(corr_post) else 0.0

    hurst_med_pre = float(hurst_pre['hurst'].median()) if len(hurst_pre) else 0.5
    hurst_med_post = float(hurst_post['hurst'].median()) if len(hurst_post) else 0.5

    return {
        'Pair': f"{ty}/{tx}",
        'Coint_Frac_Pre': coint_frac_pre,
        'Coint_Frac_Post': coint_frac_post,
        'Corr_Mean_Pre': corr_mean_pre,
        'Corr_Mean_Post': corr_mean_post,
        'Hurst_Med_Pre': hurst_med_pre,
        'Hurst_Med_Post': hurst_med_post,
    }


def save_table(summary):
    table_path = os.path.join(TABLES_DIR, 'table11_failure_modes.tex')

    def fmt(x):
        return f"{x:.2f}"

    with open(table_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Failure-mode summary across 60 pairs (rolling 126-day window).}\n")
        f.write("\\label{tab:failure_modes}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Pre-2020} & \\textbf{Post-2020} & \\textbf{$\\Delta$} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Cointegration fraction & {fmt(summary['Coint_Frac_Pre'])} & {fmt(summary['Coint_Frac_Post'])} & {fmt(summary['Coint_Frac_Post'] - summary['Coint_Frac_Pre'])} \\\\\n")
        f.write(f"Mean return correlation & {fmt(summary['Corr_Mean_Pre'])} & {fmt(summary['Corr_Mean_Post'])} & {fmt(summary['Corr_Mean_Post'] - summary['Corr_Mean_Pre'])} \\\\\n")
        f.write(f"Median Hurst exponent & {fmt(summary['Hurst_Med_Pre'])} & {fmt(summary['Hurst_Med_Post'])} & {fmt(summary['Hurst_Med_Post'] - summary['Hurst_Med_Pre'])} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  ✓ Saved {table_path}")


def save_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    axes[0].boxplot([df['Coint_Frac_Pre'], df['Coint_Frac_Post']], labels=['Pre-2020', 'Post-2020'])
    axes[0].set_title('Cointegration Fraction')
    axes[0].set_ylabel('Fraction')

    axes[1].boxplot([df['Corr_Mean_Pre'], df['Corr_Mean_Post']], labels=['Pre-2020', 'Post-2020'])
    axes[1].set_title('Mean Return Correlation')

    axes[2].boxplot([df['Hurst_Med_Pre'], df['Hurst_Med_Post']], labels=['Pre-2020', 'Post-2020'])
    axes[2].set_title('Median Hurst Exponent')

    for ax in axes:
        ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = os.path.join(FIGURES_DIR, f'fig21_failure_modes.{ext}')
        fig.savefig(out_path)
    plt.close(fig)

    print(f"  ✓ Saved {os.path.join(FIGURES_DIR, 'fig21_failure_modes.pdf/png')}")


def main():
    print("\n" + "=" * 70)
    print("  DAY 6: FAILURE-MODE ANALYSIS (FULL SCALE)")
    print("=" * 70)

    pairs = load_pairs()
    if not pairs:
        raise RuntimeError("No pairs found in expanded_universe_summary.csv")

    rows = []
    for i, (ty, tx) in enumerate(pairs, 1):
        print(f"  [{i:02d}/{len(pairs)}] {ty}/{tx}")
        rows.append(summarize_pair(ty, tx))

    df = pd.DataFrame(rows)
    pair_out = os.path.join(RESULTS_DIR, 'failure_mode_pair_stats.csv')
    df.to_csv(pair_out, index=False)
    print(f"\n  ✓ Saved {pair_out}")

    summary = {
        'Coint_Frac_Pre': float(df['Coint_Frac_Pre'].median()),
        'Coint_Frac_Post': float(df['Coint_Frac_Post'].median()),
        'Corr_Mean_Pre': float(df['Corr_Mean_Pre'].median()),
        'Corr_Mean_Post': float(df['Corr_Mean_Post'].median()),
        'Hurst_Med_Pre': float(df['Hurst_Med_Pre'].median()),
        'Hurst_Med_Post': float(df['Hurst_Med_Post'].median()),
    }

    summary_df = pd.DataFrame([summary])
    summary_out = os.path.join(RESULTS_DIR, 'failure_mode_summary.csv')
    summary_df.to_csv(summary_out, index=False)
    print(f"  ✓ Saved {summary_out}")

    save_table(summary)
    save_figure(df)


if __name__ == '__main__':
    main()
