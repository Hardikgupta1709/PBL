#!/usr/bin/env python3
"""Generate risk reduction plots (paired MaxDD boxplot)."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "Research", "results", "expanded_universe_summary.csv")
FIG_DIR = os.path.join(ROOT, "Paper", "figures")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(SUMMARY)

    uf = df["UF_OOS_MaxDD_%"].abs().dropna()
    f = df["F_OOS_MaxDD_%"].abs().dropna()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([uf, f], tick_labels=["UF", "F"], showfliers=False)
    ax.set_title("OOS MaxDD |UF vs F|")
    ax.set_ylabel("|MaxDD| (%)")
    ax.grid(axis="y", alpha=0.3)

    out_png = os.path.join(FIG_DIR, "fig20_maxdd_boxplot.png")
    out_pdf = os.path.join(FIG_DIR, "fig20_maxdd_boxplot.pdf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
