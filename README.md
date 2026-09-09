# A Risk-Aware Regime-Adaptive Framework for Statistical Arbitrage Under Post-COVID Cointegration Instability

**Zenodo archive — source code, data acquisition scripts, and reproduction assets**

[![Deposit Licence: CC-BY 4.0](https://img.shields.io/badge/Deposit_Licence-CC--BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code Licence: MIT](https://img.shields.io/badge/Code_Licence-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21022049.svg)](https://doi.org/10.5281/zenodo.21022049)

---

## Overview

This archive contains the complete research codebase and reproduction scripts for:

> **"A Risk-Aware Regime-Adaptive Framework for Statistical Arbitrage
> Under Post-COVID Cointegration Instability"**
> Hardik Gupta and Rishi Gupta
> Department of Computer Science and Engineering, Manipal University Jaipur
> *Submitted to Applied Artificial Intelligence (Taylor & Francis)*

The framework couples three components into a single pipeline:

1. **Adaptive Kalman filter** with Median Absolute Deviation (MAD) gain
   attenuation for robust, time-varying hedge-ratio estimation.
2. **Three-state Random Forest regime classifier** distinguishing Normal,
   Volatile, and Crisis market states from 21 rolling features.
3. **Real-time pair-health gate** executing rolling ADF, Hurst exponent,
   and Engle-Granger diagnostics as a hard entry precondition before each trade.

Evaluated on 60 US equity pairs over a strict 2021-2025 out-of-sample window,
the pair-health gate reduces median maximum drawdown by 3.9x
(Wilcoxon p < 1e-4, N = 60) relative to the unfiltered baseline.

---

## Archive Contents

All paths are relative to the archive root. This deposit contains source
code and scripts only. Price data are downloaded at runtime from Yahoo
Finance (no login or account required — see the Data section below).
Regime classifier weights are trained in memory during each run using
fixed hyperparameters and a fixed random seed, producing deterministic
results (see the Classifier section below).

| Path | Size | Description |
|---|---|---|
| `README.md` | — | This file |
| `__init__.py` | 341 B | Package initialiser |
| `hybrid_pairs_trading.py` | 32 KB | Core trading engine: Kalman filter, regime classifier, health gate, signal logic, backtest loop, and performance attribution |
| `run_paper_trading.py` | 11 KB | Entry point for the Alpaca Markets paper-trading and rotation workflow |
| `Dockerfile` | 1 KB | Container definition for a fully reproducible execution environment |
| `requirements.txt` | — | Pinned Python dependency list (see Software Environment) |
| `Core_Strategy/` | — | Strategy modules including `conservative_strategy.py`, which implements data download, feature construction, regime classifier training, Kalman filter, health gate, and backtest engine |
| `Research/` | — | Experiment pipelines: ablation study, baselines, cointegration analysis, feature importance, regime evaluation, statistical tests, return attribution, 60-pair universe backtest, and master `reproduce.py` script |
| `Pair_Discovery/` | — | Pair screening utilities and 60-pair universe discovery pipeline |
| `Paper_Trading/` | — | Alpaca Markets integration: monthly rotation scan, eligibility logic, daily monitoring loop, and trade execution logging |

---

## Licence

**This Zenodo deposit** is released under
**Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
Full licence text: https://creativecommons.org/licenses/by/4.0/

You are free to share and adapt all materials in this deposit for any
purpose, provided you give appropriate credit, link to the licence,
and indicate if changes were made.

**The source code** additionally carries the **MIT Licence** (see `LICENSE`).
The MIT Licence is compatible with CC-BY 4.0 and applies specifically to
the `.py` files; the CC-BY 4.0 deposit licence governs the archive as a whole.

---

## Software Environment

**Python version:** 3.10.12

**Pinned library versions:**

| Library | Version | Purpose |
|---|---|---|
| pandas | 2.0.3 | Data manipulation and time-series alignment |
| numpy | 1.24.3 | Numerical computation |
| scikit-learn | 1.3.0 | Random Forest classifier and StandardScaler |
| statsmodels | 0.14.0 | ADF test and Engle-Granger cointegration |
| hurst | 0.0.5 | Hurst exponent estimation |
| torch | 2.0.1 | Dueling Double DQN reinforcement learning agent |
| yfinance | 0.2.28 | Yahoo Finance price data download |
| alpaca-trade-api | 3.0.0 | Alpaca Markets paper-trading integration |

Full pinned list (including transitive dependencies): `requirements.txt`

**Hardware used:** Apple M-series CPU (8-core, 16 GB unified memory), no GPU.
All wall-clock times reported in the manuscript were measured on this hardware.

**Random seeds (all components):**

| Component | Seed |
|---|---|
| numpy | 42 |
| torch | 42 |
| scikit-learn (random_state) | 42 |
| Monte Carlo permutation trials | 0 |
| Block bootstrap resamples | 0 |

With these seeds fixed, all results are fully deterministic.

---

## Quick Start

### Option A — Docker (recommended for exact environment reproducibility)

```bash
# Build the container image
docker build -t pairs_trading .

# Run the full reproduction suite inside the container
docker run --rm pairs_trading python Research/reproduce.py
```

### Option B — Local Python environment

**Requirements:** Python 3.10 or newer, pip.

```bash
# Step 1 — extract the archive and enter the directory
cd PBL_Run

# Step 2 — create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS and Linux
# venv\Scripts\activate         # Windows

# Step 3 — install pinned dependencies
pip install -r requirements.txt

# Step 4 — run the full reproduction suite
python Research/reproduce.py
```

**Targeted runs:**

```bash
python Research/reproduce.py --quick      # Core results only (approx. 10 min)
python Research/reproduce.py --figures    # Regenerate all manuscript figures
python Research/reproduce.py --tables     # Regenerate all manuscript tables
```

---

## Data

### Source and access

All price data are sourced from **Yahoo Finance** via the `yfinance` library
(v0.2.28). Yahoo Finance provides free, publicly accessible historical price
data. No account, login, API key, or subscription is required.

Data are downloaded automatically when you run any reproduction script.
No pre-downloaded files are included in this archive.

### Download and preprocessing pipeline

The data pipeline is implemented in `Core_Strategy/conservative_strategy.py`
starting at line 711. The pipeline performs the following steps in order:

1. Downloads each ticker separately using `yfinance` with `auto_adjust=True`,
   which returns closing prices already adjusted for splits and dividends.
2. Extracts the `Close` price series for each asset.
3. Aligns all series to common trading dates.
4. Drops rows with missing values using `dropna()`.
5. Computes returns using `pct_change()`.
6. Fills derived feature gaps with `ffill()` or `fillna(0)` in the
   feature construction and classifier path.

No bespoke cleaning is applied. There is no manual outlier removal,
winsorisation, or smoothing on the raw Yahoo Finance data.

### Tickers and date range

| Asset(s) | Role |
|---|---|
| BAC, PNC | Benchmark pair 1 (Banking) |
| WFC, MS | Benchmark pair 2 (Financial Services) |
| CVX, OXY | Benchmark pair 3 (Energy) |
| SPY | Market benchmark |
| 60-pair universe | See `Pair_Discovery/` for full ticker list |

Date range: 2015-01-01 through 2025-06-30 (downloaded 2025-07-01).

### Reproducibility note

Yahoo Finance applies retroactive split and dividend adjustments after the
original download date. If you re-download on a later date, minor numerical
differences from the manuscript figures are possible. For exact bitwise
reproduction, use the included Docker image, which pins the full software
environment, although it cannot pin the Yahoo Finance data itself.

---

## Regime Classifier

There are no pre-trained `.pkl` model weight files in this deposit.
The Random Forest regime classifier is trained entirely in memory during
each run. Because all random seeds are fixed (see Software Environment),
training is fully deterministic and produces identical weights on every run.

### Classifier location

Implemented in `Core_Strategy/conservative_strategy.py` starting at line 137.

### Training pipeline (three steps)

**Step 1 — Feature construction.**
Builds 21 rolling features from the price series and market index:
rolling realised volatilities (5, 10, 20, 40, 60 days), first and second
differences of rolling volatility, four rolling maximum drawdown measures,
three rolling return windows, price trend strength, range expansion,
pair return correlation and its one-day change, pair-level spread volatility,
and a binary spike indicator.

**Step 2 — Label construction.**
Assigns one of three ordinal regime labels to each training day based on
rolling 20-day realised volatility quantiles computed on the training period:

| Label | Value | Condition |
|---|---|---|
| Crisis | 0 | volatility > 90th percentile |
| Volatile | 1 | volatility > 70th percentile |
| Normal | 2 | otherwise |

**Step 3 — Model fitting.**
Fits a `StandardScaler` followed by a `RandomForestClassifier` with:

| Hyperparameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| min_samples_split | 40 |
| min_samples_leaf | 20 |
| class_weight | balanced |
| random_state | 42 |

The classifier is trained exclusively on data up to 2020-12-31 and frozen.
No test-period data (2021-2025) is used during training or recalibration.

---

## Reproducing Manuscript Results

The master reproduction script (`Research/reproduce.py`) runs all
experiments and writes all outputs in the order tables and figures
appear in the manuscript.

| Script | Manuscript output |
|---|---|
| `Research/reproduce.py` | All tables and figures (full pipeline) |
| `Research/baselines.py` | Table 2: six baselines vs proposed method |
| `Research/ablation_study.py` | Table 9: ablation across eight configurations |
| `Research/feature_analysis.py` | Table 10 and Figure 7: feature importance |
| `Research/statistical_tests.py` | Tables 6, 7, 8: bootstrap CI, Monte Carlo, Wilcoxon |
| `Core_Strategy/strategy_validator.py` | Tables 4, 5 and Figure 10: walk-forward analysis |
| `Research/cointegration_analysis.py` | Figure 6 and Table 12: rolling diagnostics |
| `Research/regime_evaluation.py` | Table 3 and Figure 2: classifier comparison |
| `Research/attribution.py` | Table 8 and Figure 9: return attribution |
| `Research/universe_backtest.py` | Figures 13, 14, 19, 20: 60-pair universe results |

**Output locations:**

| Location | Contents |
|---|---|
| `Research/results/` | Intermediate CSV and JSON files |
| `Research/figures/` | All manuscript figures in PDF and PNG |
| `Research/tables/` | All manuscript tables in CSV |

**Expected run times on Apple M-series CPU (single core, no GPU):**

| Task | Time |
|---|---|
| Single-pair backtest | ~1 s |
| Pair health monitoring (per pair) | ~3 s |
| Dynamic filter (per pair) | ~6 s |
| RL training (10 episodes) | ~16 s |
| 10-pair universe backtest | ~57 s |
| Full pipeline (10 pairs) | ~4 min |

---

## Paper-Trading Workflow

The Alpaca Markets paper-trading workflow requires API credentials.
These are not included in this archive. Obtain a free paper-trading
account at https://app.alpaca.markets

Set credentials as environment variables before running:

```bash
export ALPACA_KEY_ID="your_key_id"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

Start the daily monitoring loop:

```bash
python run_paper_trading.py
```

This runs the pipeline described in Section 5.12 of the manuscript
(Deployment Feasibility Study). The monthly rotation scan across 38 pairs
completes in approximately 2 minutes on a single CPU.


---

## Contact

**Hardik Gupta** — code, data, and reproduction queries
hardik.2427030615@muj.manipal.edu

**Rishi Gupta** — manuscript and research queries
rishi.gupta@jaipur.manipal.edu

Department of Computer Science and Engineering
Manipal University Jaipur, Jaipur, Rajasthan 302026, India

---

## Acknowledgements

The authors thank the Department of Computer Science and Engineering,
Manipal University Jaipur, for access to computational resources.
The open-source communities behind scikit-learn, statsmodels, PyTorch,
and yfinance are gratefully acknowledged.