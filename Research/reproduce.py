#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Reproducibility: lock all random seeds before any imports that use them
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reproduce")

# Directories
RESULTS_DIR = PROJECT_ROOT / "Research" / "results"
FIGURES_DIR = PROJECT_ROOT / "Paper" / "figures"
TABLES_DIR = PROJECT_ROOT / "Paper" / "tables"
for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters (from config_experiments.yaml)
# ---------------------------------------------------------------------------

PAIRS: List[Tuple[str, str]] = [
    ("BAC", "PNC"),
    ("WFC", "MS"),
    ("CVX", "OXY"),
]
PAIR_LABELS = ["BAC/PNC", "WFC/MS", "CVX/OXY"]

START_DATE = "2015-01-01"
END_DATE = "2025-06-30"
TRAIN_END = "2020-12-31"
MARKET_TICKER = "SPY"
SLIPPAGE_BPS = 5.0


# ---------------------------------------------------------------------------
# 1. Baseline Comparison (Improvement 4)
# ---------------------------------------------------------------------------

def run_baselines() -> None:
    """Run all 7 baselines on each pair; save comparison CSVs."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Baseline Comparison (7 methods × 3 pairs)")
    logger.info("=" * 60)
    from Research.baselines import run_all_baselines

    all_rows = []
    for (ty, tx), label in zip(PAIRS, PAIR_LABELS):
        logger.info(f"  Pair: {label}")
        try:
            comp_df, _ = run_all_baselines(
                ty, tx,
                start_date=START_DATE, end_date=END_DATE,
                train_end_date=TRAIN_END, slippage_bps=SLIPPAGE_BPS,
                verbose=False,
            )
            comp_df["Pair"] = label
            all_rows.append(comp_df)
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        out = RESULTS_DIR / "baselines_comparison.csv"
        df.to_csv(out, index=False)
        logger.info(f"  ✓ Saved {out}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# 2. Ablation Study (Improvement 5)
# ---------------------------------------------------------------------------

def run_ablation() -> None:
    """Run 9-configuration ablation on each pair."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Ablation Study (9 configs × 3 pairs)")
    logger.info("=" * 60)
    from Research.ablation_study import run_ablation_single_pair

    all_rows = []
    for (ty, tx), label in zip(PAIRS, PAIR_LABELS):
        logger.info(f"  Pair: {label}")
        try:
            abl_df = run_ablation_single_pair(
                ty, tx,
                start_date=START_DATE, end_date=END_DATE,
                train_end_date=TRAIN_END, slippage_bps=SLIPPAGE_BPS,
                verbose=False,
            )
            abl_df["Pair"] = label
            all_rows.append(abl_df)
        except Exception as e:
            logger.error(f"  FAILED: {e}")

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        out = RESULTS_DIR / "ablation_results.csv"
        df.to_csv(out, index=False)
        logger.info(f"  ✓ Saved {out}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# 3. Feature Analysis (Improvement 6)
# ---------------------------------------------------------------------------

def run_features() -> None:
    """Run SHAP + Gini + permutation importance analysis."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Feature Importance Analysis")
    logger.info("=" * 60)
    from Research.feature_analysis import run_feature_analysis

    for (ty, tx), label in zip(PAIRS, PAIR_LABELS):
        logger.info(f"  Pair: {label}")
        try:
            result = run_feature_analysis(
                ty, tx,
                start_date=START_DATE, end_date=END_DATE,
                train_end_date=TRAIN_END, verbose=False,
            )
            # Save consensus ranking
            if "consensus" in result:
                out = RESULTS_DIR / f"features_{ty}_{tx}.csv"
                result["consensus"].to_csv(out, index=False)
                logger.info(f"  ✓ Saved {out}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 4. Statistical Tests: Bootstrap + Monte Carlo (Improvements 2, 7, 8)
# ---------------------------------------------------------------------------

def run_stats(quick: bool = False) -> None:
    """Run bootstrap CIs, Monte Carlo significance, BH correction."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Statistical Tests (Bootstrap + Monte Carlo)")
    logger.info("=" * 60)
    from Research.statistical_tests import run_statistical_analysis_multi_pair

    n_bootstrap = 1000 if quick else 10000
    try:
        result = run_statistical_analysis_multi_pair(
            PAIRS,
            start_date=START_DATE, end_date=END_DATE,
            train_end_date=TRAIN_END, n_bootstrap=n_bootstrap,
            fdr_alpha=0.05, verbose=True,
        )
        # Save summary
        out = RESULTS_DIR / "statistical_tests.json"
        # Convert non-serialisable types
        serialisable = {}
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                serialisable[k] = v.to_dict(orient="records")
            elif isinstance(v, (np.integer, np.floating)):
                serialisable[k] = float(v)
            elif isinstance(v, dict):
                serialisable[k] = str(v)
            else:
                serialisable[k] = str(v)
        with open(out, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        logger.info(f"  ✓ Saved {out}")
    except Exception as e:
        logger.error(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 5. Walk-Forward Analysis (Improvement 13)
# ---------------------------------------------------------------------------

def run_walkforward() -> None:
    """Run 32-fold expanding walk-forward on each pair."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Walk-Forward Analysis (32 folds × 3 pairs)")
    logger.info("=" * 60)
    import yfinance as yf
    from Core_Strategy.strategy_validator import WalkForwardAnalyzer

    wfa = WalkForwardAnalyzer(
        min_train_days=504, step_days=63,
        test_days=126, min_folds=8,
    )

    for (ty, tx), label in zip(PAIRS, PAIR_LABELS):
        logger.info(f"  Pair: {label}")
        try:
            data = yf.download([ty, tx, MARKET_TICKER],
                               start=START_DATE, end=END_DATE,
                               auto_adjust=True)["Close"]
            stock_y = data[ty].dropna()
            stock_x = data[tx].dropna()
            market = data[MARKET_TICKER].dropna()
            idx = stock_y.index.intersection(stock_x.index).intersection(market.index)
            stock_y, stock_x, market = stock_y[idx], stock_x[idx], market[idx]

            wf_df = wfa.run(stock_y, stock_x, market,
                            slippage_bps=SLIPPAGE_BPS, verbose=False)
            out = RESULTS_DIR / f"walkforward_{ty}_{tx}.csv"
            wf_df.to_csv(out, index=False)
            logger.info(f"  ✓ Saved {out}  ({len(wf_df)} folds)")
        except Exception as e:
            logger.error(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 6. Cointegration Analysis (Improvement 9)
# ---------------------------------------------------------------------------

def run_cointegration() -> None:
    """Rolling cointegration + Hurst exponent analysis."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: Cointegration Stability Analysis")
    logger.info("=" * 60)
    from Research.cointegration_analysis import run_cointegration_multi_pair

    try:
        summary = run_cointegration_multi_pair(
            PAIRS,
            start_date=START_DATE, end_date=END_DATE,
            train_end_date=TRAIN_END, window=126, verbose=True,
        )
        out = RESULTS_DIR / "cointegration_summary.csv"
        summary.to_csv(out, index=False)
        logger.info(f"  ✓ Saved {out}")
    except Exception as e:
        logger.error(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 7. Regime Evaluation (Improvement 12)
# ---------------------------------------------------------------------------

def run_regime() -> None:
    """Compare RF vs HMM vs simple rules."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7: Regime Detection Evaluation")
    logger.info("=" * 60)
    from Research.regime_evaluation import run_regime_evaluation

    for (ty, tx), label in zip(PAIRS, PAIR_LABELS):
        logger.info(f"  Pair: {label}")
        try:
            result = run_regime_evaluation(
                ty, tx,
                start_date=START_DATE, end_date=END_DATE,
                train_end_date=TRAIN_END, slippage_bps=SLIPPAGE_BPS,
                verbose=False,
            )
            if "summary" in result and result["summary"] is not None:
                out = RESULTS_DIR / f"regime_eval_{ty}_{tx}.csv"
                if isinstance(result["summary"], pd.DataFrame):
                    result["summary"].to_csv(out, index=False)
                else:
                    pd.DataFrame([result["summary"]]).to_csv(out, index=False)
                logger.info(f"  ✓ Saved {out}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 8. Performance Attribution (Improvement 11)
# ---------------------------------------------------------------------------

def run_attribution() -> None:
    """Decompose OOS returns into 4 alpha sources."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 8: Performance Attribution")
    logger.info("=" * 60)
    from Research.attribution import run_attribution_multi_pair

    try:
        attr_df = run_attribution_multi_pair(
            PAIRS,
            start_date=START_DATE, end_date=END_DATE,
            train_end_date=TRAIN_END, slippage_bps=SLIPPAGE_BPS,
            verbose=True,
        )
        out = RESULTS_DIR / "attribution_results.csv"
        attr_df.to_csv(out, index=False)
        logger.info(f"  ✓ Saved {out}")
    except Exception as e:
        logger.error(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 9. Generate All Figures (Improvement 17)
# ---------------------------------------------------------------------------

def run_figures() -> None:
    """Generate all 11 camera-ready figures."""
    logger.info("=" * 60)
    logger.info("GENERATING: All 11 Paper Figures")
    logger.info("=" * 60)
    from Research.generate_paper_figures import (
        fig1_architecture, fig2_regime_bands, fig3_ablation,
        fig4_baselines, fig5_sensitivity, fig6_cointegration,
        fig7_feature_importance, fig8_cumulative_returns,
        fig9_attribution, fig10_walkforward, fig_bonus_multipair,
    )

    figs = [
        ("fig1_architecture", fig1_architecture),
        ("fig2_regime_bands", fig2_regime_bands),
        ("fig3_ablation", fig3_ablation),
        ("fig4_baselines", fig4_baselines),
        ("fig5_sensitivity", fig5_sensitivity),
        ("fig6_cointegration", fig6_cointegration),
        ("fig7_feature_importance", fig7_feature_importance),
        ("fig8_cumulative_returns", fig8_cumulative_returns),
        ("fig9_attribution", fig9_attribution),
        ("fig10_walkforward", fig10_walkforward),
        ("fig_bonus_multipair", fig_bonus_multipair),
    ]
    for name, func in figs:
        try:
            func()
            logger.info(f"  ✓ {name}")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")


# ---------------------------------------------------------------------------
# 10. Generate All Tables (Improvement 18)
# ---------------------------------------------------------------------------

def run_tables() -> None:
    """Generate all 8 standalone LaTeX tables."""
    logger.info("=" * 60)
    logger.info("GENERATING: All 8 Paper Tables")
    logger.info("=" * 60)
    from Research.generate_paper_tables import main as generate_all_tables
    generate_all_tables()


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------

def run_all(quick: bool = False) -> None:
    """Execute the complete reproduction pipeline."""
    t0 = time.time()

    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║  Regime-Adaptive Pairs Trading — Full Reproduction          ║
║  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                              ║
║  Mode: {'QUICK' if quick else 'FULL '}                                            ║
║  Seed: {RANDOM_SEED}                                                ║
╚══════════════════════════════════════════════════════════════╝
"""
    logger.info(banner)

    # Phase 1: Experiments (generate results CSVs)
    logger.info("PHASE 1: Running Experiments")
    logger.info("-" * 60)

    run_baselines()
    run_ablation()
    run_features()
    run_stats(quick=quick)
    run_walkforward()
    run_cointegration()
    run_regime()
    run_attribution()

    # Phase 2: Generate paper assets
    logger.info("\nPHASE 2: Generating Paper Assets")
    logger.info("-" * 60)

    run_figures()
    run_tables()

    # Summary
    elapsed = time.time() - t0
    n_csv = len(list(RESULTS_DIR.glob("*.csv")))
    n_json = len(list(RESULTS_DIR.glob("*.json")))
    n_fig = len(list(FIGURES_DIR.glob("*.pdf")))
    n_tab = len(list(TABLES_DIR.glob("*.tex")))

    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  REPRODUCTION COMPLETE                                      ║
║  Elapsed: {elapsed/60:.1f} minutes                                     ║
║  Results: {n_csv} CSV + {n_json} JSON in Research/results/             ║
║  Figures: {n_fig} PDFs in Paper/figures/                               ║
║  Tables:  {n_tab} LaTeX files in Paper/tables/                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    logger.info(summary)

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed": RANDOM_SEED,
        "pairs": PAIR_LABELS,
        "train_end": TRAIN_END,
        "elapsed_seconds": round(elapsed, 1),
        "results_csv": n_csv,
        "results_json": n_json,
        "figures_pdf": n_fig,
        "tables_tex": n_tab,
        "mode": "quick" if quick else "full",
    }
    manifest_path = RESULTS_DIR / "reproduction_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all experiments for the pairs trading paper.",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer Monte Carlo trials (1k vs 10k)")
    parser.add_argument("--tables", action="store_true",
                        help="Generate tables only")
    parser.add_argument("--figures", action="store_true",
                        help="Generate figures only")
    parser.add_argument("--experiments", action="store_true",
                        help="Run experiments only (no figures/tables)")
    args = parser.parse_args()

    if args.tables:
        run_tables()
    elif args.figures:
        run_figures()
    elif args.experiments:
        run_baselines()
        run_ablation()
        run_features()
        run_stats(quick=args.quick)
        run_walkforward()
        run_cointegration()
        run_regime()
        run_attribution()
    else:
        run_all(quick=args.quick)


if __name__ == "__main__":
    main()
