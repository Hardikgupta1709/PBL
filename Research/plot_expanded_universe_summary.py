#!/usr/bin/env python3
"""Generate summary distribution plots for expanded universe results."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "Research", "results", "expanded_universe_summary.csv")
FIG_DIR = os.path.join(ROOT, "Paper", "figures")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(df["UF_OOS_Sharpe"].dropna(), bins=20, alpha=0.6, label="UF")
    axes[0].hist(df["F_OOS_Sharpe"].dropna(), bins=20, alpha=0.6, label="F")
    axes[0].axvline(df["UF_OOS_Sharpe"].median(), color="tab:blue", linestyle="--")
    axes[0].axvline(df["F_OOS_Sharpe"].median(), color="tab:orange", linestyle="--")
    axes[0].set_title("OOS Sharpe Distribution")
    axes[0].set_xlabel("Sharpe")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(df["UF_OOS_MaxDD_%"].abs().dropna(), bins=20, alpha=0.6, label="UF")
    axes[1].hist(df["F_OOS_MaxDD_%"].abs().dropna(), bins=20, alpha=0.6, label="F")
    axes[1].axvline(df["UF_OOS_MaxDD_%"].abs().median(), color="tab:blue", linestyle="--")
    axes[1].axvline(df["F_OOS_MaxDD_%"].abs().median(), color="tab:orange", linestyle="--")
    axes[1].set_title("OOS MaxDD Distribution")
    axes[1].set_xlabel("|MaxDD| (%)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "fig19_universe_distributions.png")
    out_pdf = os.path.join(FIG_DIR, "fig19_universe_distributions.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
