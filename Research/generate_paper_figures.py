"""
Paper Figure Generator — Publication-Quality Figures (300 DPI)
================================================================
Week 5: Generate all 10 figures for the research paper.

Figures:
    1. System Architecture Diagram (flowchart)
    2. Regime Classification Example (coloured regime bands)
    3. Ablation Bar Chart (Sharpe contribution per component)
    4. Baseline Comparison Bar Chart (our method vs 6 baselines)
    5. Parameter Sensitivity Heatmap (entry_z × exit_z)
    6. Cointegration Stability Plot (rolling p-value + trades)
    7. Feature Importance Bar Chart (SHAP top-10)
    8. Cumulative Return Curves (our method vs baselines)
    9. Performance Attribution Bar (return decomposition)
   10. Walk-Forward Fold Results (per-fold Sharpe bar chart)

All figures saved to Paper/figures/ as PDF + PNG (300 DPI).
Matplotlib + seaborn, consistent colour palette, LaTeX-safe labels.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, StrictRegimeClassifier, ConservativeKalman,
    calculate_zscore, download_data
)
from Core_Strategy.strategy_validator import (
    WalkForwardAnalyzer, MonteCarloValidator, BootstrapAnalyzer, SensitivityAnalyzer
)

# ── Style ──
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
})

COLORS = {
    'ours': '#2196F3',
    'baseline': '#9E9E9E',
    'positive': '#4CAF50',
    'negative': '#F44336',
    'accent': '#FF9800',
    'normal': '#4CAF50',
    'volatile': '#FF9800',
    'crisis': '#F44336',
    'market': '#9C27B0',
    'palette': ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548'],
}

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'Paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

TRAIN_END = '2020-12-31'
PAIRS = [('BAC', 'PNC'), ('WFC', 'MS'), ('CVX', 'OXY')]


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {name}.pdf / .png")


# =====================================================================
# FIGURE 1: SYSTEM ARCHITECTURE DIAGRAM
# =====================================================================
def fig1_architecture():
    print("\n  [Fig 1] System Architecture Diagram")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    boxes = [
        # (x, y, w, h, label, color)
        (0.3, 5.5, 2.0, 1.0, 'Price Data\n(Y, X, Market)', '#E3F2FD'),
        (3.0, 5.5, 2.0, 1.0, 'Adaptive Kalman\nFilter (MAD)', '#BBDEFB'),
        (6.0, 5.5, 2.0, 1.0, 'Dynamic Hedge\nRatio & Spread', '#90CAF9'),
        (0.3, 3.5, 2.0, 1.0, 'Feature\nEngineering', '#E8F5E9'),
        (3.0, 3.5, 2.0, 1.0, 'RF Regime\nClassifier', '#C8E6C9'),
        (6.0, 3.5, 2.0, 1.0, 'Regime Label\n(N/V/C)', '#A5D6A7'),
        (1.5, 1.5, 2.0, 1.0, 'Z-Score Signal\nGeneration', '#FFF3E0'),
        (4.5, 1.5, 2.0, 1.0, 'Regime-Gated\nExecution', '#FFE0B2'),
        (7.5, 1.5, 2.0, 1.0, 'Position & Risk\nManagement', '#FFCCBC'),
    ]

    for (x, y, w, h, label, color) in boxes:
        fancy = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(fancy)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#212121')

    # Arrows
    arrow_kw = dict(arrowstyle='->', color='#555', lw=1.5,
                    connectionstyle='arc3,rad=0')
    pairs = [
        ((2.3, 6.0), (3.0, 6.0)),
        ((5.0, 6.0), (6.0, 6.0)),
        ((1.3, 5.5), (1.3, 4.5)),
        ((2.3, 4.0), (3.0, 4.0)),
        ((5.0, 4.0), (6.0, 4.0)),
        ((7.0, 5.5), (3.5, 2.5)),
        ((7.0, 3.5), (5.5, 2.5)),
        ((3.5, 1.5), (3.5, 2.5)),
        ((3.5, 2.0), (4.5, 2.0)),
        ((6.5, 2.0), (7.5, 2.0)),
    ]
    for (start, end) in pairs:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    ax.set_title('Regime-Adaptive Pairs Trading Framework', fontsize=14, fontweight='bold', pad=15)
    save_fig(fig, 'fig1_architecture')


# =====================================================================
# FIGURE 2: REGIME CLASSIFICATION EXAMPLE
# =====================================================================
def fig2_regime_bands():
    print("  [Fig 2] Regime Classification Example")
    stock_y, stock_x, market = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')
    system = ConservativeSystem()
    results = system.run_backtest(stock_y, stock_x, market,
                                  train_end_date=TRAIN_END, verbose=False)
    oos = results[results['period'] == 'oos'].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Top: Spread with regime colouring
    spread = oos['spread'].values
    dates = oos.index
    regimes = oos['regime'].values

    ax1.plot(dates, spread, color='#333', lw=0.8, alpha=0.8)
    regime_colors = {2: COLORS['normal'], 1: COLORS['volatile'], 0: COLORS['crisis']}
    for r_val, r_col in regime_colors.items():
        mask = regimes == r_val
        ax1.fill_between(dates, spread.min(), spread.max(), where=mask,
                         alpha=0.15, color=r_col, linewidth=0)

    ax1.set_ylabel('Spread')
    ax1.set_title('BAC/PNC Spread with RF Regime Classification (OOS: 2021-2025)', fontweight='bold')
    ax1.legend(handles=[
        mpatches.Patch(color=COLORS['normal'], alpha=0.3, label='Normal'),
        mpatches.Patch(color=COLORS['volatile'], alpha=0.3, label='Volatile'),
        mpatches.Patch(color=COLORS['crisis'], alpha=0.3, label='Crisis'),
    ], loc='upper right', framealpha=0.9)

    # Bottom: Z-score with entry/exit thresholds
    zscore = oos['z_score'].values
    ax2.plot(dates, zscore, color='#2196F3', lw=0.8)
    ax2.axhline(1.5, color=COLORS['positive'], ls='--', lw=0.8, label='Entry (1.5)')
    ax2.axhline(-1.5, color=COLORS['positive'], ls='--', lw=0.8)
    ax2.axhline(0.5, color=COLORS['accent'], ls=':', lw=0.8, label='Exit (0.5)')
    ax2.axhline(-0.5, color=COLORS['accent'], ls=':', lw=0.8)
    ax2.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax2.set_ylabel('Z-Score')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    save_fig(fig, 'fig2_regime_bands')


# =====================================================================
# FIGURE 3: ABLATION BAR CHART
# =====================================================================
def fig3_ablation():
    print("  [Fig 3] Ablation Study Bar Chart")
    from Research.ablation_study import ABLATION_CONFIGS, compute_oos_metrics

    stock_y, stock_x, market = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')

    names = []
    sharpes = []
    for name, run_fn in ABLATION_CONFIGS.items():
        try:
            results = run_fn(stock_y, stock_x, market, TRAIN_END, 5.0)
            metrics = compute_oos_metrics(results)
            names.append(name)
            sharpes.append(metrics.get('Sharpe', 0.0))
        except Exception as e:
            print(f"    ⚠️ {name} failed: {e}")
            names.append(name)
            sharpes.append(0.0)

    full_sharpe = sharpes[0]
    deltas = [s - full_sharpe for s in sharpes]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    colors_bar = [COLORS['ours'] if i == 0 else
                  (COLORS['positive'] if deltas[i] >= 0 else COLORS['negative'])
                  for i in range(len(names))]
    bars = ax.bar(x, sharpes, color=colors_bar, edgecolor='white', linewidth=0.8)

    ax.axhline(full_sharpe, color='#333', ls='--', lw=1, alpha=0.5, label=f'Full System ({full_sharpe:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('OOS Sharpe Ratio')
    ax.set_title('Ablation Study: BAC/PNC (OOS Performance When Removing Each Component)', fontweight='bold')
    ax.legend(loc='upper right')

    # Annotate deltas
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        if i == 0:
            continue
        ax.annotate(f'{delta:+.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=7, color='#555')

    plt.tight_layout()
    save_fig(fig, 'fig3_ablation')


# =====================================================================
# FIGURE 4: BASELINE COMPARISON BAR CHART
# =====================================================================
def fig4_baselines():
    print("  [Fig 4] Baseline Comparison Bar Chart")
    from Research.baselines import run_all_baselines

    comp_df, all_results = run_all_baselines('BAC', 'PNC',
                                    train_end_date=TRAIN_END, verbose=False)

    methods = comp_df['Method'].tolist()
    sharpes = comp_df['Sharpe'].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    colors_bar = [COLORS['ours']] + [COLORS['baseline']] * (len(methods) - 1)
    bars = ax.barh(x, sharpes, color=colors_bar, edgecolor='white', linewidth=0.8, height=0.6)

    ax.set_yticks(x)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('OOS Sharpe Ratio (Net of Costs)')
    ax.set_title('Method Comparison: BAC/PNC Out-of-Sample Performance', fontweight='bold')
    ax.axvline(0, color='#333', lw=0.5, alpha=0.5)

    for bar, s in zip(bars, sharpes):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{s:.3f}', va='center', fontsize=8, color='#333')

    plt.tight_layout()
    save_fig(fig, 'fig4_baselines')


# =====================================================================
# FIGURE 5: PARAMETER SENSITIVITY HEATMAP
# =====================================================================
def fig5_sensitivity():
    print("  [Fig 5] Sensitivity Heatmap")
    stock_y, stock_x, market = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')

    entry_vals = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    exit_vals = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9]

    grid = np.zeros((len(exit_vals), len(entry_vals)))

    for i, exit_z in enumerate(exit_vals):
        for j, entry_z in enumerate(entry_vals):
            if exit_z >= entry_z:
                grid[i, j] = np.nan
                continue
            system = ConservativeSystem()
            results = system.run_backtest(
                stock_y, stock_x, market,
                train_end_date=TRAIN_END,
                entry_z_normal=entry_z, exit_z_normal=exit_z,
                slippage_bps=5.0, verbose=False,
            )
            oos = results[results['period'] == 'oos']
            ret = oos['strategy_return'].dropna()
            if len(ret) > 0 and ret.std() > 0:
                tr = (1 + ret).prod() - 1
                ar = (1 + tr) ** (252 / len(ret)) - 1
                grid[i, j] = ar / (ret.std() * np.sqrt(252))
            else:
                grid[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = np.isnan(grid)
    vabs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.5)
    sns.heatmap(grid, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=[f'{v:.2f}' for v in entry_vals],
                yticklabels=[f'{v:.1f}' for v in exit_vals],
                mask=mask, ax=ax, vmin=-vabs, vmax=vabs,
                cbar_kws={'label': 'Sharpe Ratio'})
    ax.set_xlabel('Entry Z-Score Threshold')
    ax.set_ylabel('Exit Z-Score Threshold')
    ax.set_title('Parameter Sensitivity: BAC/PNC OOS Sharpe Ratio', fontweight='bold')

    profitable = np.nansum(grid > 0)
    total_valid = np.nansum(~np.isnan(grid))
    pct = profitable / total_valid * 100 if total_valid > 0 else 0
    ax.text(0.02, -0.08, f'Profitable region: {pct:.0f}% of parameter space',
            transform=ax.transAxes, fontsize=9, color='#555')

    plt.tight_layout()
    save_fig(fig, 'fig5_sensitivity')


# =====================================================================
# FIGURE 6: COINTEGRATION STABILITY PLOT
# =====================================================================
def fig6_cointegration():
    print("  [Fig 6] Cointegration Stability Plot")
    from Research.cointegration_analysis import (
        rolling_cointegration, rolling_hurst, rolling_half_life
    )

    stock_y, stock_x, market = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')

    coint_df = rolling_cointegration(stock_y, stock_x, window=126, step=5)
    spread = stock_y - stock_x  # simplified spread for Hurst
    hurst_df = rolling_hurst(spread, window=126, step=5)

    train_end = pd.Timestamp(TRAIN_END)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: cointegration p-value
    ax1.plot(coint_df.index, coint_df['coint_pvalue'], color='#2196F3', lw=1)
    ax1.axhline(0.05, color=COLORS['positive'], ls='--', lw=1, label='p = 0.05 threshold')
    ax1.axvline(train_end, color='#333', ls=':', lw=1.2, label='Train / OOS Split')
    ax1.fill_between(coint_df.index, 0, coint_df['coint_pvalue'],
                     where=coint_df['coint_pvalue'] < 0.05, alpha=0.2,
                     color=COLORS['positive'], label='Cointegrated')
    ax1.fill_between(coint_df.index, 0, coint_df['coint_pvalue'],
                     where=coint_df['coint_pvalue'] >= 0.05, alpha=0.15,
                     color=COLORS['negative'], label='Not Cointegrated')
    ax1.set_ylabel('Cointegration p-value')
    ax1.set_title('BAC/PNC: Rolling Cointegration & Mean-Reversion Stability', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 1)

    # Bottom: Hurst exponent
    ax2.plot(hurst_df.index, hurst_df['hurst'], color='#FF9800', lw=1)
    ax2.axhline(0.5, color='#333', ls='--', lw=1, label='H = 0.5 (random walk)')
    ax2.axvline(train_end, color='#333', ls=':', lw=1.2)
    ax2.fill_between(hurst_df.index, 0, hurst_df['hurst'],
                     where=hurst_df['hurst'] < 0.5, alpha=0.2,
                     color=COLORS['positive'], label='Mean-Reverting (H < 0.5)')
    ax2.fill_between(hurst_df.index, 0.5, hurst_df['hurst'],
                     where=hurst_df['hurst'] >= 0.5, alpha=0.15,
                     color=COLORS['negative'], label='Trending (H >= 0.5)')
    ax2.set_ylabel('Hurst Exponent')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    save_fig(fig, 'fig6_cointegration')


# =====================================================================
# FIGURE 7: FEATURE IMPORTANCE (SHAP TOP-10)
# =====================================================================
def fig7_feature_importance():
    print("  [Fig 7] Feature Importance (SHAP)")
    from Research.feature_analysis import prepare_classifier_data, shap_analysis

    classifier, features, labels, train_mask, test_mask = prepare_classifier_data(
        'BAC', 'PNC', train_end_date=TRAIN_END
    )
    shap_df = shap_analysis(classifier, features, test_mask, max_samples=500, verbose=False)

    if len(shap_df) == 0:
        print("  ⚠️ SHAP unavailable, skipping Fig 7")
        return

    top10 = shap_df.head(10).iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(top10))
    ax.barh(y_pos, top10['Mean_Abs_SHAP'].values, color=COLORS['ours'],
            edgecolor='white', linewidth=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10['Feature'].values, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('RF Regime Classifier: Top-10 Feature Importance (SHAP)', fontweight='bold')

    for i, v in enumerate(top10['Mean_Abs_SHAP'].values):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8, color='#555')

    plt.tight_layout()
    save_fig(fig, 'fig7_feature_importance')


# =====================================================================
# FIGURE 8: CUMULATIVE RETURN CURVES
# =====================================================================
def fig8_cumulative_returns():
    print("  [Fig 8] Cumulative Return Curves")
    from Research.baselines import run_all_baselines

    comp_df, all_results = run_all_baselines('BAC', 'PNC',
                                    train_end_date=TRAIN_END, verbose=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    method_names = {'full_system': 'Full System (Ours)', 'gatev': 'Gatev Distance',
                    'ols_coint': 'OLS Cointegration', 'kalman_only': 'Kalman-Only',
                    'regime_only': 'Regime-Only (OLS)', 'bh_stocks': 'B&H Stocks',
                    'spy': 'SPY B&H'}
    for idx, (key, res) in enumerate(all_results.items()):
        if 'oos_returns' not in res:
            continue
        oos_ret = res['oos_returns']
        cum_ret = (1 + oos_ret).cumprod()
        lw = 2.0 if idx == 0 else 1.0
        alpha_val = 1.0 if idx == 0 else 0.6
        color = COLORS['palette'][idx % len(COLORS['palette'])]
        ax.plot(cum_ret.index, cum_ret.values, label=method_names.get(key, key),
                color=color, lw=lw, alpha=alpha_val)

    ax.axhline(1.0, color='#333', ls=':', lw=0.8, alpha=0.4)
    ax.set_ylabel('Cumulative Return (1 = initial)')
    ax.set_xlabel('Date')
    ax.set_title('BAC/PNC: OOS Cumulative Returns — Our Method vs Baselines', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    save_fig(fig, 'fig8_cumulative_returns')


# =====================================================================
# FIGURE 9: PERFORMANCE ATTRIBUTION BAR
# =====================================================================
def fig9_attribution():
    print("  [Fig 9] Performance Attribution")
    from Research.attribution import (
        compute_regime_timing_alpha, compute_mean_reversion_alpha,
        compute_position_sizing_alpha
    )

    stock_y, stock_x, market = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')

    regime = compute_regime_timing_alpha(stock_y, stock_x, market, TRAIN_END)
    mr = compute_mean_reversion_alpha(stock_y, stock_x, market, TRAIN_END)
    ps = compute_position_sizing_alpha(stock_y, stock_x, market, TRAIN_END)

    sources = {
        'Regime Timing': regime['regime_timing_alpha_%'],
        'Mean Reversion': mr['mean_reversion_alpha_%'],
        'Position Sizing': ps['sizing_alpha_%'],
    }
    # Residual
    total_ret = regime['total_return_with_regime_%']
    explained = sum(sources.values())
    sources['Residual'] = round(total_ret - explained, 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(sources.keys())
    vals = list(sources.values())
    colors_bar = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in vals]
    bars = ax.bar(names, vals, color=colors_bar, edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='#333', ls='-', lw=0.5)
    ax.axhline(total_ret, color=COLORS['ours'], ls='--', lw=1.2,
               label=f'Total OOS Return: {total_ret:.1f}%')

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:+.1f}%', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9)

    ax.set_ylabel('Return Contribution (%)')
    ax.set_title('BAC/PNC: OOS Performance Attribution', fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    save_fig(fig, 'fig9_attribution')


# =====================================================================
# FIGURE 10: WALK-FORWARD FOLD RESULTS
# =====================================================================
def fig10_walkforward():
    print("  [Fig 10] Walk-Forward Fold Results")
    stock_y, stock_x, market = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')

    wf = WalkForwardAnalyzer(min_train_days=504, step_days=63, test_days=126)
    wf_results = wf.run(stock_y, stock_x, market, verbose=False)

    if len(wf_results) == 0:
        print("  ⚠️ Walk-forward returned no folds")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})

    n_folds = len(wf_results)
    x = np.arange(n_folds)
    sharpes = wf_results['Sharpe'].values
    returns = wf_results['Return_%'].values

    colors_bar = [COLORS['positive'] if s >= 0 else COLORS['negative'] for s in sharpes]
    ax1.bar(x, sharpes, color=colors_bar, edgecolor='white', linewidth=0.5, width=0.7)
    ax1.axhline(0, color='#333', ls='-', lw=0.5)
    avg_sh = sharpes.mean()
    ax1.axhline(avg_sh, color=COLORS['ours'], ls='--', lw=1.2,
                label=f'Mean Sharpe: {avg_sh:.3f}')

    # Trend line
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, sharpes)
    z = np.polyfit(x, sharpes, 1)
    ax1.plot(x, np.polyval(z, x), color=COLORS['accent'], ls='--', lw=1,
             label=f'Trend: rho={rho:.3f} (p={pval:.3f})')

    ax1.set_ylabel('OOS Sharpe Ratio')
    ax1.set_title(f'BAC/PNC: Walk-Forward Analysis ({n_folds} Expanding Folds)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)

    # Bottom: returns
    colors_ret = [COLORS['positive'] if r >= 0 else COLORS['negative'] for r in returns]
    ax2.bar(x, returns, color=colors_ret, edgecolor='white', linewidth=0.5, width=0.7)
    ax2.axhline(0, color='#333', ls='-', lw=0.5)
    consistency = (returns > 0).mean() * 100
    ax2.axhline(returns.mean(), color=COLORS['ours'], ls='--', lw=1,
                label=f'Mean: {returns.mean():.2f}% | Consistency: {consistency:.0f}%')
    ax2.set_ylabel('OOS Return (%)')
    ax2.set_xlabel('Fold Number')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xticks(x[::4])

    plt.tight_layout()
    save_fig(fig, 'fig10_walkforward')


# =====================================================================
# MULTI-PAIR SUMMARY FIGURE (BONUS)
# =====================================================================
def fig_bonus_multipair():
    print("  [Bonus] Multi-Pair Summary")
    pair_data = {}
    for ty, tx in PAIRS:
        stock_y, stock_x, market = download_data(ty, tx, 'SPY', '2015-01-01', '2025-06-30')
        system = ConservativeSystem()
        results = system.run_backtest(stock_y, stock_x, market,
                                      train_end_date=TRAIN_END,
                                      slippage_bps=5.0, verbose=False)
        oos = results[results['period'] == 'oos']
        ret = oos['strategy_return'].dropna()
        baseline_ret = oos['baseline_return'].dropna()

        def sharpe(r):
            if len(r) == 0 or r.std() == 0:
                return 0.0
            tr = (1 + r).prod() - 1
            ar = (1 + tr) ** (252 / len(r)) - 1
            return ar / (r.std() * np.sqrt(252))

        pair_data[f'{ty}/{tx}'] = {
            'Ours': sharpe(ret),
            'No Regime': sharpe(baseline_ret),
        }

    fig, ax = plt.subplots(figsize=(8, 4))
    pairs_names = list(pair_data.keys())
    x = np.arange(len(pairs_names))
    width = 0.35

    ours_vals = [pair_data[p]['Ours'] for p in pairs_names]
    noreg_vals = [pair_data[p]['No Regime'] for p in pairs_names]

    ax.bar(x - width / 2, ours_vals, width, label='Full System', color=COLORS['ours'],
           edgecolor='white')
    ax.bar(x + width / 2, noreg_vals, width, label='No Regime Gating', color=COLORS['baseline'],
           edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(pairs_names)
    ax.set_ylabel('OOS Sharpe Ratio (Net)')
    ax.set_title('Multi-Pair Comparison: Full System vs No Regime Gating', fontweight='bold')
    ax.axhline(0, color='#333', ls='-', lw=0.5)
    ax.legend()

    for i, (o, n) in enumerate(zip(ours_vals, noreg_vals)):
        delta = o - n
        ax.text(i, max(o, n) + 0.02, f'Delta: {delta:+.3f}', ha='center', fontsize=8, color='#555')

    plt.tight_layout()
    save_fig(fig, 'fig_bonus_multipair')


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  PAPER FIGURE GENERATOR — 10 Publication-Quality Figures")
    print(f"  Output: {FIGDIR}")
    print("=" * 70)

    fig1_architecture()
    fig2_regime_bands()
    fig3_ablation()
    fig4_baselines()
    fig5_sensitivity()
    fig6_cointegration()
    fig7_feature_importance()
    fig8_cumulative_returns()
    fig9_attribution()
    fig10_walkforward()
    fig_bonus_multipair()

    print(f"\n  {'=' * 50}")
    figs = [f for f in os.listdir(FIGDIR) if f.endswith('.pdf')]
    print(f"  Done! {len(figs)} PDF figures saved to Paper/figures/")
    print(f"  {'=' * 50}")
