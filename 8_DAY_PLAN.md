# 8-Day Plan: From 5.5/10 to 8/10

Purpose: Execute a focused, day-by-day upgrade path to raise the paper and framework score to a solid 8/10. This plan is detailed and sequential to avoid missing critical steps.

Success criteria by Day 8:
- Expanded-universe results (>= 50 pairs) reported with summary stats and plots.
- Risk-reduction significance (MaxDD, CVaR, tail-risk) supported by proper tests.
- At least one concrete innovation implemented and evaluated (not just a reframing).
- 5-10 success pairs with enough effective sample size for credible inference.
- Paper reframed to emphasize robust evaluation + negative results + validated risk reduction.
- Clear operational feasibility section with capacity and cost break-even analysis.

Non-negotiable quality gates (must pass for an 8/10):
- >= 5 robust positive pairs with effective sample size >= 300 trading days OR healthy-day fraction >= 10%.
- At least one innovation with statistically meaningful impact (p < 0.05 on MaxDD or CVaR improvement vs baseline).
- At least one significance result that survives multiple-testing correction (BH at 5%).
- Expanded-universe evidence that risk reduction is consistent across sectors (>= 70% of pairs improved MaxDD).

---

## Day 1: Reframe the Thesis and Lock the Story

Goal: Align the paper and experiments with a strong, defensible contribution.

Tasks:
1. Reframe the core contribution as:
   - A robust framework that quantifies when pairs trading fails post-2020.
   - A validated risk-reduction system (health filter) rather than pure alpha generation.
2. Identify and document 3-4 new paper claims that are empirically verifiable:
   - Example: "Health filter reduces tail risk significantly across a broad universe." 
   - Example: "Pairs trading becomes viable only under strict health gating in specific sectors."
3. Update the paper outline (sections and subsection titles):
   - Add a "Failure Mode Analysis" subsection in Discussion.
   - Add a "Risk Reduction Significance" subsection in Results.
   - Add a "Operational Feasibility" subsection.
4. Map new results to figure/table placeholders.

Deliverables:
- Updated section outline in notes (or a short plan doc).
- Clear list of experimental claims to validate during Days 2-7.

---

## Day 2: Expanded Universe Backtest (>= 50 Pairs) + Robust Pair Discovery

Goal: Remove the "3 pairs only" criticism with broad cross-sectional evidence.

Tasks:
1. Finalize universe selection:
   - 75-100 stocks across 10 sectors.
   - Run auto_find_pairs to identify >= 50 pairs with discovery score >= threshold.
2. Add a robustness screen for candidate pairs:
   - Minimum cointegration stability in train period.
   - Minimum liquidity proxy (ADV threshold).
   - Exclude pairs with extreme spread volatility.
3. Run expanded universe backtest with identical parameters.
4. Compute summary statistics:
   - Mean/median Sharpe
   - Mean/median MaxDD
   - % positive Sharpe
   - % profitable pairs
   - Distribution of healthy-day fractions
5. Save detailed pair-level CSV results.

Deliverables:
- CSV: expanded universe pair-level results.
- Summary table and one plot (Sharpe distribution + MaxDD distribution).

---

## Day 3: Risk Reduction Significance Testing + Multiple-Testing Control

Goal: Statistically prove the health filter reduces drawdown and tail risk.

Tasks:
1. Add tests on the expanded universe:
   - Wilcoxon signed-rank test for MaxDD improvement.
   - KS test for return tails (filtered vs unfiltered).
   - CVaR (95%) reduction analysis.
2. Apply Benjamini-Hochberg correction across all pair-level tests.
3. Report effect sizes (median reduction factor, percent improvement).
4. Create a table summarizing risk metrics (UF vs F).
5. Add a paired bar or boxplot figure for MaxDD or CVaR.

Deliverables:
- Updated results CSV with UF/F risk metrics.
- Table and figure for the paper.

---

## Day 4: Implement One Real Innovation + Measurable Ablation Target

Goal: Raise innovation score by adding one distinct, defensible mechanism.

Pick ONE (and only one) innovation to implement fully:
A) Regime-adaptive transaction costs (costs rise in crisis).
B) Health-aware position sizing (size = f(health score)).
C) Bayesian regime gating (replace RF labels with probabilistic gate).

Tasks:
1. Implement the chosen innovation in Core_Strategy.
2. Define a measurable target before running ablation:
   - Example target: reduce median MaxDD by >= 15% vs baseline.
   - Example target: improve CVaR(95%) by >= 10%.
3. Run ablation to show its isolated effect.
4. Produce a short analysis: does it improve MaxDD, Sharpe, or stability?

Deliverables:
- Code change + ablation results.
- One short figure or table showing impact.

---

## Day 5: Success Pair Validation With Adequate Sample Size + Effective Sample Check

Goal: Identify 5-10 pairs that show genuine OOS stability.

Tasks:
1. From the expanded universe, select pairs with:
   - Positive Sharpe after costs.
   - >= 300 effective trading days OR >= 10% healthy-day fraction.
2. If < 5 pairs meet the criteria, expand universe threshold or add sectors.
3. Run full pipeline on these pairs:
   - Walk-forward, Monte Carlo, bootstrap.
4. Summarize results in a "success pairs" table.

Deliverables:
- Success pairs table with CI and MC p-values.
- One figure with equity curves for top 2-3 pairs.

---

## Day 6: Failure Mode Analysis + Market Structure Explanation

Goal: Strengthen impact by explaining why pairs trading fails post-2020.

Tasks:
1. Add a short empirical analysis:
   - Rolling correlation decay.
   - Cointegration fraction by period (pre-2020 vs post-2020).
   - Volatility regime shifts.
2. Draft a Discussion subsection:
   - Liquidity regime shifts, ETF flows, retail pressure, faster mean reversion decay.
3. Connect this to the negative results.

Deliverables:
- One additional figure or table.
- Draft discussion text (1-2 pages).

---

## Day 7: Operational Feasibility + Capacity

Goal: Strengthen "real-world" viability even if alpha is small.

Tasks:
1. Expand break-even cost analysis:
   - Range of one-way costs (2-30 bps).
   - Sensitivity curves for top success pairs.
2. Add capacity estimates:
   - Max AUM before impact erodes half the edge.
3. Add operational constraints:
   - Avg trade frequency, median holding period, fraction of time in market.

Deliverables:
- Updated economic significance table.
- One figure showing cost sensitivity.

---

## Day 8: Paper Integration and Final Polish

Goal: Merge all improvements into a coherent, high-score paper.

Tasks:
1. Update Results section with new tables and figures.
2. Update Discussion section with failure-mode analysis.
3. Update Contributions section to reflect actual findings.
4. Ensure all claims are backed by results.
5. Run full reproduction to validate everything.

---

## Days 9-10 (Full-Scale Robustness Layer)

Goal: Add a clear “reviewer-proof” robustness layer focused on external validity,
cost realism, and sensitivity checks. These are executed even if prior gates pass,
to reduce rejection risk and strengthen generalisability claims.

Day 9: Cross-Market Robustness + Expanded Universe
- Expand universe to 150–200 stocks and re-run auto_find_pairs at full scale.
- Add sector-level constraints to avoid unstable sectors and concentration.
- Add a second universe (e.g., EU large-caps or ETFs) with the same pipeline.
- Re-run expanded backtest + risk reduction tests and update success-pairs list.

Day 9 To-Do Checklist:
[ ] Expand universe list and re-run auto_find_pairs
[ ] Add sector constraints and re-run discovery
[ ] Build second-market universe and run full pipeline
[ ] Update expanded-universe summary CSVs and plots
[ ] Update success-pairs table with new markets

Day 10: Cost Realism + Sensitivity + Stress
- Add slippage/impact sensitivity curves (3–4 cost levels) for top pairs.
- Add health-threshold sensitivity grid (ADF/Hurst/EG) and report stability bands.
- Add alternative bootstrap (block size variants) for key pairs.
- Re-run ablation with updated robustness settings and re-integrate results.

Day 10 To-Do Checklist:
[ ] Run cost-sensitivity analysis and update economic-significance figure
[ ] Run health-threshold sensitivity grid and summarize stability
[ ] Run bootstrap block-size variants and update CI narrative
[ ] Re-run ablation with robustness settings and update table
[ ] Update paper discussion with robustness outcomes

Deliverables (Days 9–10):
- Updated paper with cross-market results and sensitivity analysis integrated.
- Robustness appendix or supplementary note (if space constrained).
- Final reproducibility run log.

---

## Tracking Checklist (Daily)

Each day, confirm:
- Data saved with timestamp.
- Figures generated and saved in Paper/figures/.
- Tables saved in Paper/tables/.
- Summary notes updated for the day.
- Paper placeholders updated if relevant.

---

## If Anything Slips

Priority order if time becomes tight:
1. Expanded universe + risk significance (Days 2-3)
2. One innovation + ablation (Day 4)
3. Success pairs with proper sample size (Day 5)
4. Paper integration (Day 8)

If the quality gates are not met by Day 8, immediately activate Days 9-10.

This order preserves the strongest impact on score improvement.
