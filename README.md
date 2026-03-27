# Regime-Adaptive Pairs Trading with Robust Kalman Filtering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Reproducible](https://img.shields.io/badge/Reproducible-Docker-blue?style=flat-square&logo=docker)

**A statistically rigorous pairs trading framework combining adaptive Kalman filters  
with random-forest regime detection, evaluated on US equities (2015–2025).**

[Paper](#paper) · [Quick Start](#quick-start) · [Reproduce](#reproduction) · [Results](#key-results) · [Citation](#citation)

</div>

---

## Abstract

We present a pairs trading system that augments classical cointegration-based
spread trading with two innovations: (1) an adaptive Kalman filter whose
process noise is modulated by a random-forest regime classifier trained on
macro-financial features, and (2) a sensitivity-derived entry/exit band
mechanism that widens thresholds in non-mean-reverting regimes. We evaluate
the system on three US equity pairs (BAC/PNC, WFC/MS, CVX/OXY) using a
strict temporal split (train ≤ 2020, OOS 2021–2025) and 32-fold expanding
walk-forward analysis. Honest out-of-sample results show negative Sharpe
ratios for all pairs, consistent with the known difficulty of achieving
post-cost alpha in mature statistical arbitrage markets. Regime gating
improves BAC/PNC by +19.5% cumulative return relative to always-on trading.
A comprehensive ablation study (9 configurations) and 7-baseline comparison
contextualise the contribution of each component.

---

## Paper

The full paper is in `Paper/main.tex` (IEEE conference format, ~10 pages).
Pre-built figures (11 PDFs) are in `Paper/figures/`, and 8 generated LaTeX
tables are in `Paper/tables/`.

To compile on **Overleaf**, upload `Paper/overleaf_upload.zip` as a new project.

---

## Key Results

| Pair | OOS Sharpe | Max DD | Regime Gating Δ |
|------|-----------|--------|-----------------|
| BAC/PNC | −0.126 | −12.3% | +19.5% cumulative |
| WFC/MS | −0.045 | −8.7% | +6.2% cumulative |
| CVX/OXY | −0.613 | −18.1% | +3.8% cumulative |

All OOS Sharpe ratios are negative after realistic transaction costs (5 bps
slippage). See the paper for full discussion of why this is expected and what
the ablation/attribution analysis reveals about component contributions.

---

## Project Structure

```
PBL_Run/
├── Core_Strategy/              # Production strategy code
│   ├── conservative_strategy.py    # Regime-adaptive Kalman pairs trader
│   └── strategy_validator.py       # Walk-forward + bootstrap validation
├── Research/                   # Experiment & analysis modules
│   ├── reproduce.py                # ★ One-command full reproduction
│   ├── baselines.py                # 7-baseline comparison
│   ├── ablation_study.py           # 9-config ablation
│   ├── feature_analysis.py         # SHAP + Gini + permutation
│   ├── statistical_tests.py        # Bootstrap CIs + Monte Carlo
│   ├── cointegration_analysis.py   # Rolling ADF + Hurst
│   ├── regime_evaluation.py        # RF vs HMM vs simple rules
│   ├── attribution.py              # 4-source return decomposition
│   ├── universe_backtest.py        # Multi-pair OOS backtest
│   ├── generate_paper_figures.py   # All 11 figures
│   ├── generate_paper_tables.py    # All 8 tables
│   ├── config_experiments.yaml     # Centralised parameters
│   └── results/                    # Intermediate CSVs/JSONs
├── Paper/                      # IEEE LaTeX manuscript
│   ├── main.tex                    # Full paper (~760 lines)
│   ├── references.bib              # 37 BibTeX entries
│   ├── figures/                    # 11 camera-ready figures
│   ├── tables/                     # 8 standalone .tex tables
│   └── overleaf_upload.zip         # Ready-to-upload archive
├── Pair_Discovery/             # Automated pair screening
│   └── auto_find_pairs.py         # Cointegration scanner
├── Paper_Trading/              # Live paper trading (Alpaca)
│   ├── alpaca_paper_trader.py
│   ├── Monitoring_Dashboard.py     # Streamlit dashboard
│   └── config.py
├── Configuration/              # Legacy config
├── Dockerfile                  # Reproducibility container
├── requirements.txt            # Pinned dependencies (Python 3.12.9)
├── EXPERIMENTS.md              # Per-experiment reproduction guide
└── RESEARCH_UPGRADE_PLAN.md    # 7-week upgrade roadmap
```

---

## Quick Start

### Prerequisites

- Python 3.12+ (tested on 3.12.9)
- ~4 GB RAM
- Internet connection (for `yfinance` data download)

### Installation

```bash
git clone <repo-url> && cd PBL_Run

# Option A: venv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Option B: Docker (fully self-contained)
docker build -t pairs-trading .
docker run --rm pairs-trading              # full reproduction
docker run --rm pairs-trading python Research/reproduce.py --quick
```

### Run the Strategy (single pair)

```python
from Core_Strategy.conservative_strategy import ConservativePairsStrategy

strategy = ConservativePairsStrategy(
    ticker_y="BAC", ticker_x="PNC",
    lookback=252, entry_z=2.0, exit_z=0.5,
    regime_filter=True,
)
results = strategy.run(start_date="2015-01-01", end_date="2025-06-30")
print(results["oos_sharpe"])
```

---

## Reproduction

A single command reproduces **all** experiments, figures, and tables:

```bash
python Research/reproduce.py           # Full run (~30-45 min)
python Research/reproduce.py --quick   # Reduced Monte Carlo (~5-10 min)
python Research/reproduce.py --figures # Figures only
python Research/reproduce.py --tables  # Tables only
```

**Outputs:**
- `Research/results/` — CSVs and JSONs for every experiment
- `Paper/figures/` — 11 camera-ready PDF + PNG figures
- `Paper/tables/` — 8 standalone LaTeX table files
- `Research/results/reproduction_manifest.json` — Provenance metadata

**Determinism:** Random seed 42 is locked at startup. Results depend on
`yfinance` data (which may shift slightly with provider updates), but the
qualitative conclusions are robust.

See [EXPERIMENTS.md](EXPERIMENTS.md) for per-experiment details.

---

## Experiments

| # | Experiment | Module | Output |
|---|-----------|--------|--------|
| 1 | Baseline comparison (7 methods) | `Research/baselines.py` | Table 2, Fig 4 |
| 2 | Ablation study (9 configs) | `Research/ablation_study.py` | Table 3, Fig 3 |
| 3 | Feature importance (SHAP) | `Research/feature_analysis.py` | Table 5, Fig 7 |
| 4 | Statistical tests (bootstrap) | `Research/statistical_tests.py` | Table 1 CIs |
| 5 | Walk-forward (32 folds) | `Core_Strategy/strategy_validator.py` | Table 4, Fig 10 |
| 6 | Cointegration stability | `Research/cointegration_analysis.py` | Fig 6 |
| 7 | Regime detection eval | `Research/regime_evaluation.py` | Table 7 |
| 8 | Performance attribution | `Research/attribution.py` | Table 8, Fig 9 |

---

## Configuration

All experiment parameters are centralised in
[`Research/config_experiments.yaml`](Research/config_experiments.yaml):

```yaml
temporal_split:
  train_end: "2020-12-31"
strategy_defaults:
  entry_z: 2.0
  exit_z: 0.5
  lookback: 252
reproducibility:
  random_seed: 42
  n_bootstrap: 10000
```

---

## Paper Trading

The system supports live paper trading via the Alpaca API:

```bash
# Set API keys in Paper_Trading/config.py
python Paper_Trading/alpaca_paper_trader.py

# Monitor via Streamlit dashboard
streamlit run Paper_Trading/Monitoring_Dashboard.py
```

---

## Dependencies

Key packages (all pinned in `requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.3.5 | Numerical computation |
| pandas | 2.3.3 | Data manipulation |
| scipy | 1.16.3 | Statistical tests |
| scikit-learn | 1.7.2 | Random-forest regime classifier |
| statsmodels | 0.14.5 | Cointegration (ADF, Johansen) |
| shap | 0.51.0 | Feature importance |
| yfinance | 1.1.0 | Market data |
| matplotlib | 3.10.7 | Figures |
| alpaca-py | 0.43.2 | Paper trading API |

---

## Citation

If you use this code or reference the methodology:

```bibtex
@inproceedings{pairs_trading_2025,
  title   = {Regime-Adaptive Pairs Trading with Robust Kalman Filtering:
             A Multi-Component Analysis},
  author  = {Hardik},
  year    = {2025},
  note    = {IEEE conference format}
}
```

---

## License

This project is released under the MIT License.
