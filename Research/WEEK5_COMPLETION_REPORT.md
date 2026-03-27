# Week 5 Completion Report: Paper Writing — Structure
**Date:** March 9, 2026  
**Theme:** Transform code into a story reviewers want to read

---

## Deliverables Completed

### ✅ Improvement 15: Full LaTeX Paper (`Paper/main.tex`)
- **Format:** IEEE Conference (8–10 pages)
- **Sections completed (all 8):**
  1. **Abstract** (248 words) — Problem, gap, method, results, impact
  2. **Introduction** (1.5 pages) — Three challenges + three contributions
  3. **Related Work** (1.5 pages) — 5 subsections covering classical PT, Kalman, regimes, ML, comparison table
  4. **Methodology** (2.5 pages) — Full math: Kalman filter equations (6 equations), MAD robustness, RF features, signal generation, cost model, Algorithm pseudocode
  5. **Experimental Setup** (1 page) — Data, 6 baselines, evaluation protocol (WF, MC, bootstrap, BH, ablation)
  6. **Results** (2 pages) — 10 subsections with 8 tables and 10 figure references, ALL with real numbers
  7. **Discussion** (0.5 pages) — Honest negative results, regime gating value, cointegration breakdown, 5 limitations
  8. **Conclusion** (0.5 pages) — Summary, contributions, honest framing

- **Key design decision:** Paper reports HONEST negative results (negative Sharpe, failed MC tests) and frames the contribution as the *evaluation framework* rather than a profitable strategy. This is the academically correct approach.

### ✅ Improvement 16: References (`Paper/references.bib`)
- **36 BibTeX entries** covering:
  - Pairs trading foundations: 7 (Gatev, Vidyamurthy, Do & Faff, Rad et al., Krauss et al., Liu et al.)
  - Cointegration: 3 (Engle-Granger, Johansen, Hurst)
  - Kalman filter in finance: 4 (Elliott, Triantafyllopoulos, Montana, Clegg & Krauss)
  - Regime switching: 5 (Hamilton, Ang & Timmermann, Guidolin, Bulla, Nystrup)
  - Machine learning: 6 (Breiman, Lundberg/SHAP, Gu et al., Chen et al., Sarmento, Kim et al.)
  - Statistical methodology: 6 (Efron, Ledoit-Wolf, Benjamini-Hochberg, Harvey et al., Diebold-Mariano, Cohen)
  - Transaction costs: 2 (Almgren-Chriss, Brinson)
  - Recent 2023-2025: 4 (Fil, Zhang, Wang, Kim)

### ✅ All Paper Figures Generated (`Paper/figures/`)
**11 figures × 2 formats (PDF + PNG at 300 DPI) = 22 files**

| # | Figure | File | Description |
|---|--------|------|-------------|
| 1 | System Architecture | `fig1_architecture.pdf` | Flowchart of 3 coupled subsystems |
| 2 | Regime Classification | `fig2_regime_bands.pdf` | BAC/PNC spread with coloured regime bands + z-score |
| 3 | Ablation Study | `fig3_ablation.pdf` | Bar chart: 9 configs, Sharpe delta from full system |
| 4 | Baseline Comparison | `fig4_baselines.pdf` | Horizontal bar: our method vs 6 baselines |
| 5 | Sensitivity Heatmap | `fig5_sensitivity.pdf` | Entry_z × exit_z Sharpe heatmap (7×6 grid) |
| 6 | Cointegration Stability | `fig6_cointegration.pdf` | Rolling coint p-value + Hurst exponent |
| 7 | Feature Importance | `fig7_feature_importance.pdf` | SHAP top-10 features for RF classifier |
| 8 | Cumulative Returns | `fig8_cumulative_returns.pdf` | OOS cumulative returns: ours vs baselines |
| 9 | Attribution | `fig9_attribution.pdf` | 4-source return decomposition bar chart |
| 10 | Walk-Forward | `fig10_walkforward.pdf` | 32-fold Sharpe + return bars with trend line |
| 11 | Multi-Pair | `fig_bonus_multipair.pdf` | 3-pair comparison: Full System vs No Regime |

### ✅ Build Script (`Paper/build.sh`)
- Compiles with pdflatex + bibtex (3-pass)
- Requires MacTeX: `brew install --cask mactex-no-gui`

---

## Paper Structure Summary

```
Paper/
├── main.tex              # Full IEEE paper (~550 lines)
├── references.bib        # 36 BibTeX references
├── build.sh              # Compilation script
└── figures/
    ├── fig1_architecture.pdf/png
    ├── fig2_regime_bands.pdf/png
    ├── fig3_ablation.pdf/png
    ├── fig4_baselines.pdf/png
    ├── fig5_sensitivity.pdf/png
    ├── fig6_cointegration.pdf/png
    ├── fig7_feature_importance.pdf/png
    ├── fig8_cumulative_returns.pdf/png
    ├── fig9_attribution.pdf/png
    ├── fig10_walkforward.pdf/png
    └── fig_bonus_multipair.pdf/png
```

---

## Key Metrics Used in Paper (All Real)

| Metric | BAC/PNC | WFC/MS | CVX/OXY |
|--------|---------|--------|---------|
| OOS Sharpe | −0.126 | −0.049 | −0.613 |
| OOS Return | −7.42% | −2.93% | −90.59% |
| vs No-Regime | +19.45% | −0.99% | −43.73% |
| Walk-Fwd Consistency | 50% | 38% | 31% |
| MC p-value | 0.553 | 0.484 | 0.842 |
| Bootstrap 95% CI | [−0.96, +0.90] | [−0.85, +1.03] | [−0.85, +0.62] |

---

## What Makes This Paper Academically Strong

1. **Honest negative results** — Reports true OOS performance (negative Sharpe), doesn't cherry-pick
2. **Rigorous methodology** — Walk-forward (32 folds), Monte Carlo (10k), bootstrap, BH correction
3. **Complete ablation** — 9 configs proving each component's contribution
4. **Feature comparison table** — Shows our method vs 10 prior works on 5 dimensions
5. **Live validation** — Paper trading results included (even though they're unflattering)
6. **Reproducibility** — All code released, temporal protocol documented

---

## To Compile the Paper

```bash
# Install MacTeX (one-time)
brew install --cask mactex-no-gui

# Build PDF
cd Paper && bash build.sh
```

---

## Next: Week 6 (Paper Polish + Publication-Quality Figures)
- Second-pass revision of all sections
- Consistent math notation check
- Figure/table reference verification
- Peer review integration
