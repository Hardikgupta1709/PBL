#!/usr/bin/env python3
"""Innovation ablation across full expanded universe.

Evaluates:
- Health-aware sizing vs baseline health filter (no sizing)
- RF prob gate vs RF label (no health filter)

Outputs:
  Research/results/innovation_ablation.csv
  Research/results/innovation_ablation_summary.txt
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Core_Strategy.conservative_strategy import StrictRegimeClassifier, download_data
from Research.dynamic_pair_selector import backtest_with_dynamic_filter
from Research.regime_evaluation import compute_economic_value

RESULTS_DIR = os.path.join(ROOT, "Research", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "expanded_universe_summary.csv")

TRAIN_END = "2020-12-31"
START_DATE = "2015-01-01"
END_DATE = "2025-06-30"
SLIPPAGE_BPS = 5.0
PROB_GATE = 0.50
SIZING_THRESHOLDS = (0.40, 0.65, 0.80)


def _median_delta(values_a, values_b):
    a = pd.Series(values_a).dropna()
    b = pd.Series(values_b).dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median(b) - np.median(a))


def main() -> None:
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    pairs = df["Pair"].tolist()

    rows = []

    for pair in pairs:
        try:
            ty, tx = pair.split("_")
        except ValueError:
            continue

        stock_y, stock_x, market = download_data(ty, tx, "SPY", START_DATE, END_DATE)
        if stock_y is None or stock_x is None or market is None:
            continue

        # --- Health sizing ablation (filtered vs filtered+sizing) ---
        base = backtest_with_dynamic_filter(
            stock_y, stock_x, market,
            train_end_date=TRAIN_END,
            slippage_bps=SLIPPAGE_BPS,
            apply_health_sizing=False,
            verbose=False,
        )
        sized = backtest_with_dynamic_filter(
            stock_y, stock_x, market,
            train_end_date=TRAIN_END,
            slippage_bps=SLIPPAGE_BPS,
            apply_health_sizing=True,
            sizing_thresholds=SIZING_THRESHOLDS,
            verbose=False,
        )

        base_oos = base["metrics_filtered"]["oos"]
        sized_oos = sized["metrics_filtered"]["oos"]

        # --- RF probability gate ablation (no health filter) ---
        prices_df = pd.DataFrame({"Y": stock_y, "X": stock_x}).dropna()
        clf = StrictRegimeClassifier()
        features = clf.create_features(prices_df, market)
        labels = clf.label_regimes(features)

        valid = ~(features.isna().any(axis=1) | labels.isna())
        features_clean = features[valid]
        labels_clean = labels[valid]
        train_mask = features_clean.index <= pd.Timestamp(TRAIN_END)

        clf.train(features_clean[train_mask], labels_clean[train_mask])
        rf_labels = clf.predict(features_clean)

        probs = clf.predict_proba(features_clean)
        classes = list(clf.model.classes_)
        if 2 in classes:
            normal_idx = classes.index(2)
            prob_normal = pd.Series(probs[:, normal_idx], index=features_clean.index)
        else:
            prob_normal = pd.Series(1.0, index=features_clean.index)

        rf_prob = rf_labels.copy()
        rf_prob[prob_normal < PROB_GATE] = 0

        rf_labels_full = rf_labels.reindex(stock_y.index).fillna(2).astype(int)
        rf_prob_full = rf_prob.reindex(stock_y.index).fillna(2).astype(int)

        rf_ev = compute_economic_value(
            stock_y, stock_x, market,
            rf_labels_full,
            train_end_date=TRAIN_END,
            slippage_bps=SLIPPAGE_BPS,
            label="RF",
        )
        prob_ev = compute_economic_value(
            stock_y, stock_x, market,
            rf_prob_full,
            train_end_date=TRAIN_END,
            slippage_bps=SLIPPAGE_BPS,
            label="RF Prob",
        )

        rows.append({
            "Pair": pair,
            "Sector": df.loc[df["Pair"] == pair, "Sector"].iloc[0],
            "Base_F_Sharpe": base_oos["sharpe"],
            "Sized_F_Sharpe": sized_oos["sharpe"],
            "Base_F_MaxDD": base_oos["max_dd"],
            "Sized_F_MaxDD": sized_oos["max_dd"],
            "RF_Sharpe": rf_ev["Sharpe"],
            "RFProb_Sharpe": prob_ev["Sharpe"],
            "RF_MaxDD": rf_ev["Max_DD_%"],
            "RFProb_MaxDD": prob_ev["Max_DD_%"],
        })

    out_csv = os.path.join(RESULTS_DIR, "innovation_ablation.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Summary
    r = pd.DataFrame(rows)
    summary_lines = []
    summary_lines.append("=" * 72)
    summary_lines.append("INNOVATION ABLATION SUMMARY")
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Pairs: {len(r)}")
    summary_lines.append("=" * 72)
    summary_lines.append("")

    summary_lines.append("HEALTH SIZING (Filtered vs Filtered+Sizing)")
    summary_lines.append(f"  Median Sharpe delta: {_median_delta(r['Base_F_Sharpe'], r['Sized_F_Sharpe']):+.3f}")
    summary_lines.append(f"  Median MaxDD delta: {_median_delta(r['Base_F_MaxDD'], r['Sized_F_MaxDD']):+.3f}")
    summary_lines.append("")

    summary_lines.append("RF PROB GATE (RF vs RF Prob Gate)")
    summary_lines.append(f"  Median Sharpe delta: {_median_delta(r['RF_Sharpe'], r['RFProb_Sharpe']):+.3f}")
    summary_lines.append(f"  Median MaxDD delta: {_median_delta(r['RF_MaxDD'], r['RFProb_MaxDD']):+.3f}")

    out_txt = os.path.join(RESULTS_DIR, "innovation_ablation_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
