import numpy as np
import pandas as pd
import os
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    ConservativeSystem, StrictRegimeClassifier, ConservativeKalman,
    calculate_zscore, generate_signals, download_data
)
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    f1_score, cohen_kappa_score
)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. CONFUSION MATRIX ANALYSIS
# =============================================================================

def evaluate_confusion_matrix(
    classifier: StrictRegimeClassifier,
    features: pd.DataFrame,
    true_labels: pd.Series,
    test_mask: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Compute confusion matrix and classification metrics for the RF classifier
    on the OOS (test) period.

    Note: 'true labels' are the threshold-based regime labels.  The RF is
    trained to approximate these labels, so OOS accuracy measures generalisation
    not ground truth (there is no ground truth for market regimes).
    """
    test_features = features[test_mask]
    test_labels = true_labels[test_mask]

    if len(test_features) == 0:
        return {}

    preds = classifier.predict(test_features)

    labels_list = sorted(test_labels.unique())
    regime_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}
    target_names = [regime_names.get(l, str(l)) for l in labels_list]

    cm = confusion_matrix(test_labels, preds, labels=labels_list)
    report = classification_report(
        test_labels, preds, labels=labels_list,
        target_names=target_names, output_dict=True, zero_division=0,
    )

    accuracy = accuracy_score(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(test_labels, preds)

    results = {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': round(accuracy, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'cohen_kappa': round(kappa, 4),
        'n_test': len(test_labels),
    }

    if verbose:
        print("\n  Confusion Matrix (rows=true, cols=predicted):")
        print(f"  {'':>12}", end='')
        for name in target_names:
            print(f"{name:>10}", end='')
        print()
        for i, name in enumerate(target_names):
            print(f"  {name:>12}", end='')
            for j in range(len(target_names)):
                print(f"{cm[i, j]:>10d}", end='')
            print()

        print(f"\n  Accuracy:  {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  Cohen's κ: {kappa:.4f}")

        for name in target_names:
            r = report.get(name, {})
            print(f"  {name}: precision={r.get('precision', 0):.3f} "
                  f"recall={r.get('recall', 0):.3f} f1={r.get('f1-score', 0):.3f}")

    return results


# =============================================================================
# 2. REGIME TRANSITION ANALYSIS
# =============================================================================

def analyze_regime_transitions(
    regimes: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Analyze regime transition patterns:
    - How often does the classifier switch regimes?
    - What is the average regime duration?
    - Transition probability matrix.
    """
    regimes = regimes.dropna().astype(int)
    n = len(regimes)
    regime_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}

    # Transitions
    transitions = regimes.diff().fillna(0)
    n_transitions = (transitions != 0).sum()
    transition_rate = n_transitions / max(n, 1)

    # Transition probability matrix
    labels = sorted(regimes.unique())
    n_labels = len(labels)
    trans_matrix = np.zeros((n_labels, n_labels))

    for i in range(1, len(regimes)):
        from_r = regimes.iloc[i - 1]
        to_r = regimes.iloc[i]
        from_idx = labels.index(from_r)
        to_idx = labels.index(to_r)
        trans_matrix[from_idx, to_idx] += 1

    # Normalize rows
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_matrix / row_sums

    # Regime durations
    durations = {l: [] for l in labels}
    current_regime = regimes.iloc[0]
    current_dur = 1

    for i in range(1, len(regimes)):
        if regimes.iloc[i] == current_regime:
            current_dur += 1
        else:
            durations[current_regime].append(current_dur)
            current_regime = regimes.iloc[i]
            current_dur = 1
    durations[current_regime].append(current_dur)

    avg_durations = {}
    for l in labels:
        if durations[l]:
            avg_durations[regime_names.get(l, str(l))] = round(np.mean(durations[l]), 1)
        else:
            avg_durations[regime_names.get(l, str(l))] = 0

    # Distribution
    dist = regimes.value_counts(normalize=True).sort_index()
    regime_dist = {regime_names.get(l, str(l)): round(v * 100, 1)
                   for l, v in dist.items()}

    results = {
        'n_transitions': int(n_transitions),
        'transition_rate_per_day': round(transition_rate, 4),
        'transitions_per_year': round(transition_rate * 252, 1),
        'transition_probability_matrix': trans_prob,
        'transition_labels': [regime_names.get(l, str(l)) for l in labels],
        'avg_durations_days': avg_durations,
        'regime_distribution_%': regime_dist,
        'total_days': n,
    }

    if verbose:
        print(f"\n  Transitions: {n_transitions} ({transition_rate * 252:.1f}/yr)")
        print(f"  Regime distribution: {regime_dist}")
        print(f"  Avg durations: {avg_durations}")

        print("\n  Transition Probability Matrix:")
        names = results['transition_labels']
        print(f"  {'From\\To':>12}", end='')
        for name in names:
            print(f"{name:>10}", end='')
        print()
        for i, name in enumerate(names):
            print(f"  {name:>12}", end='')
            for j in range(len(names)):
                print(f"{trans_prob[i, j]:>10.3f}", end='')
            print()

    return results


# =============================================================================
# 3. HIDDEN MARKOV MODEL (HMM) BASELINE
# =============================================================================

class HMMRegimeDetector:
    """
    Gaussian HMM for regime detection.

    Uses market returns + volatility as observations.
    Maps HMM hidden states to CRISIS/VOLATILE/NORMAL via
    volatility-based sorting of emission means.
    """

    def __init__(self, n_regimes: int = 3, n_iter: int = 100, random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.state_mapping = {}

    def fit(self, returns: pd.Series, volatility: pd.Series) -> None:
        """Fit HMM on training data."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            # Fallback: simple Gaussian mixture approach
            self._fit_fallback(returns, volatility)
            return

        df = pd.concat([returns, volatility], axis=1).dropna()
        if df.empty:
            self._fit_fallback(returns, volatility)
            return

        obs = np.column_stack([
            df.iloc[:, 0].values,
            df.iloc[:, 1].values,
        ])

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(obs)

        # Map states by volatility level
        vol_means = self.model.means_[:, 1]  # volatility component
        sorted_states = np.argsort(vol_means)
        self.state_mapping = {
            sorted_states[0]: 2,  # lowest vol → NORMAL
            sorted_states[1]: 1,  # mid vol → VOLATILE
            sorted_states[2]: 0,  # highest vol → CRISIS
        }

    def _fit_fallback(self, returns: pd.Series, volatility: pd.Series):
        """Fallback when hmmlearn is not installed — use volatility quantiles."""
        vol = volatility.dropna()
        self._vol_q33 = vol.quantile(0.33)
        self._vol_q66 = vol.quantile(0.66)
        self.model = 'fallback'

    def predict(self, returns: pd.Series, volatility: pd.Series) -> pd.Series:
        """Predict regime states."""
        if self.model == 'fallback':
            return self._predict_fallback(volatility)

        df = pd.concat([returns, volatility], axis=1).dropna()
        if df.empty:
            return self._predict_fallback(volatility)

        obs = np.column_stack([
            df.iloc[:, 0].values,
            df.iloc[:, 1].values,
        ])
        raw_states = self.model.predict(obs)
        mapped = np.array([self.state_mapping.get(s, 2) for s in raw_states])
        return pd.Series(mapped, index=df.index)

    def _predict_fallback(self, volatility: pd.Series) -> pd.Series:
        """Quantile-based fallback prediction."""
        regimes = pd.Series(2, index=volatility.index)
        regimes[volatility > self._vol_q33] = 1
        regimes[volatility > self._vol_q66] = 0
        return regimes


# =============================================================================
# 4. SIMPLE VOLATILITY-THRESHOLD RULES
# =============================================================================

class SimpleVolRegimeDetector:
    """
    Simplest possible regime detector: fixed volatility thresholds.

    CRISIS:   20d annualized vol > vol_crisis  (default 35%)
    VOLATILE: 20d annualized vol > vol_volatile (default 22%)
    NORMAL:   otherwise
    """

    def __init__(self, vol_crisis: float = 0.35, vol_volatile: float = 0.22):
        self.vol_crisis = vol_crisis
        self.vol_volatile = vol_volatile

    def predict(self, market_returns: pd.Series) -> pd.Series:
        vol_20 = market_returns.rolling(20).std() * np.sqrt(252)
        regimes = pd.Series(2, index=market_returns.index)
        regimes[vol_20 > self.vol_volatile] = 1
        regimes[vol_20 > self.vol_crisis] = 0
        return regimes.fillna(2).astype(int)


# =============================================================================
# 5. ECONOMIC VALUE COMPARISON
# =============================================================================

def compute_economic_value(
    stock_y: pd.Series, stock_x: pd.Series, market_index: pd.Series,
    regime_series: pd.Series,
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    label: str = 'method',
) -> dict:
    """
    Run a backtest using externally-provided regime predictions.

    This allows fair economic comparison: same Kalman, same signals,
    only the regime input differs.
    """
    data = pd.DataFrame({
        'Y': stock_y, 'X': stock_x, 'Market': market_index
    }).dropna()
    train_end = pd.Timestamp(train_end_date)

    # Kalman filter
    kalman = ConservativeKalman()
    spreads, hedge_ratios = [], []
    regime_map = {0: 'crisis', 1: 'volatile', 2: 'normal'}

    for i in range(len(data)):
        r_val = int(regime_series.reindex(data.index).fillna(2).iloc[i])
        regime_str = regime_map.get(r_val, 'normal')
        et, _ = kalman.update(data['Y'].iloc[i], data['X'].iloc[i], regime=regime_str)
        spreads.append(et)
        hedge_ratios.append(kalman.get_hedge_ratio())

    data['spread'] = spreads
    data['hedge_ratio'] = hedge_ratios
    data['z_score'] = calculate_zscore(pd.Series(spreads, index=data.index), 40)

    # Align regimes
    data['regime'] = regime_series.reindex(data.index).fillna(2).astype(int)

    data['final_signal'] = generate_signals(
        data['z_score'], data['regime'],
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=2.0, exit_z_volatile=0.3,
        min_hold_days=3, max_hold_days=30, stop_loss_mult=2.0,
    )

    data['returns_Y'] = data['Y'].pct_change()
    data['returns_X'] = data['X'].pct_change()
    data['spread_return'] = data['returns_Y'] - data['hedge_ratio'] * data['returns_X']
    data['strategy_return_gross'] = data['final_signal'].shift(1) * data['spread_return']
    cost = data['final_signal'].diff().abs().fillna(0) * (slippage_bps / 10_000)
    data['strategy_return'] = data['strategy_return_gross'] - cost

    # OOS metrics
    oos = data[data.index > train_end]
    ret = oos['strategy_return'].dropna()

    if len(ret) == 0 or ret.std() == 0:
        return {'method': label, 'Sharpe': 0, 'Total_Return_%': 0,
                'Max_DD_%': 0, 'Trades': 0}

    tr = (1 + ret).prod() - 1
    n = len(ret)
    ar = (1 + tr) ** (252 / n) - 1
    av = ret.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    cum = (1 + ret).cumprod()
    dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    trades = (oos['final_signal'].diff().abs() > 0).sum() // 2

    return {
        'method': label,
        'Sharpe': round(sh, 4),
        'Total_Return_%': round(tr * 100, 2),
        'Ann_Return_%': round(ar * 100, 2),
        'Ann_Vol_%': round(av * 100, 2),
        'Max_DD_%': round(dd * 100, 2),
        'Win_Rate_%': round((ret > 0).mean() * 100, 1),
        'Trades': int(trades),
    }


# =============================================================================
# 6. FULL REGIME EVALUATION
# =============================================================================

def run_regime_evaluation(
    ticker_y: str, ticker_x: str,
    start_date: str = '2015-01-01',
    end_date: str = '2025-06-30',
    train_end_date: str = '2020-12-31',
    slippage_bps: float = 5.0,
    prob_gate_threshold: float = 0.6,
    verbose: bool = True,
) -> dict:
    """
    Complete regime detection evaluation.

    Compares RF classifier against HMM and simple vol-threshold rules,
    measuring both classification quality and economic impact.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  REGIME DETECTION EVALUATION: {ticker_y}/{ticker_x}")
        print(f"{'=' * 70}")

    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )
    data = pd.DataFrame({
        'Y': stock_y, 'X': stock_x, 'Market': market
    }).dropna()

    train_end = pd.Timestamp(train_end_date)
    train_mask = data.index <= train_end
    test_mask = ~train_mask

    market_returns = data['Market'].pct_change()

    evaluation = {}

    # -------------------------------------------------------------------------
    # A) Train RF classifier
    # -------------------------------------------------------------------------
    if verbose:
        print("\n  [A] RF Classifier...")

    rf_classifier = StrictRegimeClassifier()
    prices_df = pd.DataFrame({'Y': data['Y'], 'X': data['X']})
    features = rf_classifier.create_features(prices_df, data['Market'])
    labels = rf_classifier.label_regimes(features)

    valid = ~(features.isna().any(axis=1) | labels.isna())
    features_clean = features[valid]
    labels_clean = labels[valid]
    train_mask_clean = train_mask[valid]
    test_mask_clean = ~train_mask_clean

    rf_classifier.train(features_clean[train_mask_clean], labels_clean[train_mask_clean])
    rf_regimes = rf_classifier.predict(features_clean)
    rf_regimes_full = rf_regimes.reindex(data.index).fillna(2).astype(int)

    # RF probability-gated regimes (Bayesian-style gate on P(NORMAL))
    rf_probs = rf_classifier.predict_proba(features_clean)
    classes = list(rf_classifier.model.classes_)
    if 2 in classes:
        normal_idx = classes.index(2)
        prob_normal = pd.Series(rf_probs[:, normal_idx], index=features_clean.index)
    else:
        prob_normal = pd.Series(1.0, index=features_clean.index)

    rf_regimes_prob = rf_regimes.copy()
    rf_regimes_prob[prob_normal < prob_gate_threshold] = 0
    rf_regimes_prob_full = rf_regimes_prob.reindex(data.index).fillna(2).astype(int)

    # Confusion matrix for RF
    if verbose:
        print("\n  [A.1] RF Confusion Matrix (OOS):")
    evaluation['rf_confusion'] = evaluate_confusion_matrix(
        rf_classifier, features_clean, labels_clean, test_mask_clean, verbose=verbose
    )

    # Transition analysis for RF
    if verbose:
        print("\n  [A.2] RF Transition Analysis (OOS):")
    evaluation['rf_transitions'] = analyze_regime_transitions(
        rf_regimes_full[test_mask], verbose=verbose
    )

    # -------------------------------------------------------------------------
    # B) HMM baseline
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [B] HMM Baseline...")

    vol_20 = market_returns.rolling(20).std() * np.sqrt(252)
    hmm = HMMRegimeDetector(n_regimes=3)
    hmm.fit(
        market_returns[train_mask].dropna(),
        vol_20[train_mask].dropna(),
    )
    hmm_regimes = hmm.predict(market_returns, vol_20)
    hmm_regimes = hmm_regimes.reindex(data.index).fillna(2).astype(int)

    if verbose:
        print(f"\n  [B.1] HMM Transition Analysis (OOS):")
    evaluation['hmm_transitions'] = analyze_regime_transitions(
        hmm_regimes[test_mask], verbose=verbose
    )

    # HMM agreement with threshold labels
    hmm_oos = hmm_regimes[valid][test_mask_clean]
    true_oos = labels_clean[test_mask_clean]
    hmm_acc = accuracy_score(true_oos, hmm_oos)
    hmm_kappa = cohen_kappa_score(true_oos, hmm_oos)
    if verbose:
        print(f"  HMM vs threshold labels: Accuracy={hmm_acc:.4f}, κ={hmm_kappa:.4f}")
    evaluation['hmm_accuracy'] = round(hmm_acc, 4)
    evaluation['hmm_kappa'] = round(hmm_kappa, 4)

    # -------------------------------------------------------------------------
    # C) Simple volatility threshold
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [C] Simple Volatility-Threshold Rules...")

    simple = SimpleVolRegimeDetector()
    simple_regimes = simple.predict(market_returns)
    simple_regimes = simple_regimes.reindex(data.index).fillna(2).astype(int)

    if verbose:
        print(f"\n  [C.1] Simple Threshold Transition Analysis (OOS):")
    evaluation['simple_transitions'] = analyze_regime_transitions(
        simple_regimes[test_mask], verbose=verbose
    )

    simple_oos = simple_regimes[valid][test_mask_clean]
    simple_acc = accuracy_score(true_oos, simple_oos)
    simple_kappa = cohen_kappa_score(true_oos, simple_oos)
    if verbose:
        print(f"  Simple vs threshold labels: Accuracy={simple_acc:.4f}, κ={simple_kappa:.4f}")
    evaluation['simple_accuracy'] = round(simple_acc, 4)
    evaluation['simple_kappa'] = round(simple_kappa, 4)

    # -------------------------------------------------------------------------
    # D) No regime (always NORMAL)
    # -------------------------------------------------------------------------
    no_regime = pd.Series(2, index=data.index)

    # -------------------------------------------------------------------------
    # E) Economic value comparison
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [D] Economic Value Comparison...")

    methods = {
        'RF Classifier': rf_regimes_full,
        'RF Prob Gate': rf_regimes_prob_full,
        'HMM (3-state)': hmm_regimes,
        'Simple Vol Threshold': simple_regimes,
        'No Regime (Always Normal)': no_regime,
    }

    economic_results = []
    for method_name, regime_series in methods.items():
        ev = compute_economic_value(
            stock_y, stock_x, market,
            regime_series, train_end_date, slippage_bps,
            label=method_name,
        )
        economic_results.append(ev)

    econ_df = pd.DataFrame(economic_results)
    evaluation['economic_comparison'] = econ_df

    if verbose:
        print(f"\n  Economic Value (OOS):")
        print(f"  {'Method':<28} {'Sharpe':>8} {'Return%':>10} {'MaxDD%':>8} {'Trades':>7}")
        print(f"  {'-' * 63}")
        for _, row in econ_df.iterrows():
            print(f"  {row['method']:<28} {row['Sharpe']:>8.3f} "
                  f"{row['Total_Return_%']:>+10.2f} "
                  f"{row['Max_DD_%']:>8.2f} {row['Trades']:>7d}")

        # Find best method
        best_idx = econ_df['Sharpe'].idxmax()
        best = econ_df.loc[best_idx, 'method']
        if verbose:
            if 'RF' in best:
                print(f"\n  ✅ RF classifier achieves best Sharpe")
            else:
                print(f"\n  ⚠️ {best} outperforms RF on this pair")

    # -------------------------------------------------------------------------
    # F) RF Feature Importance (for regime)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n  [E] RF Top Features for Regime Detection:")

    fi = pd.Series(
        rf_classifier.model.feature_importances_,
        index=rf_classifier.feature_names,
    ).sort_values(ascending=False)

    evaluation['feature_importance'] = fi.to_dict()
    if verbose:
        for feat, imp in fi.head(10).items():
            bar = "█" * int(imp * 100)
            print(f"    {feat:.<25} {imp:.4f} {bar}")

    # -------------------------------------------------------------------------
    # G) Summary
    # -------------------------------------------------------------------------
    evaluation['summary'] = {
        'rf_accuracy': evaluation['rf_confusion'].get('accuracy', 0),
        'rf_f1_macro': evaluation['rf_confusion'].get('f1_macro', 0),
        'rf_kappa': evaluation['rf_confusion'].get('cohen_kappa', 0),
        'hmm_accuracy': evaluation.get('hmm_accuracy', 0),
        'hmm_kappa': evaluation.get('hmm_kappa', 0),
        'simple_accuracy': evaluation.get('simple_accuracy', 0),
        'simple_kappa': evaluation.get('simple_kappa', 0),
        'rf_sharpe': float(econ_df.loc[econ_df['method'] == 'RF Classifier', 'Sharpe'].iloc[0]) if len(econ_df) > 0 else 0,
        'hmm_sharpe': float(econ_df.loc[econ_df['method'] == 'HMM (3-state)', 'Sharpe'].iloc[0]) if len(econ_df) > 0 else 0,
        'simple_sharpe': float(econ_df.loc[econ_df['method'] == 'Simple Vol Threshold', 'Sharpe'].iloc[0]) if len(econ_df) > 0 else 0,
        'no_regime_sharpe': float(econ_df.loc[econ_df['method'] == 'No Regime (Always Normal)', 'Sharpe'].iloc[0]) if len(econ_df) > 0 else 0,
    }

    # Save chart
    try:
        _save_regime_comparison_chart(econ_df, ticker_y, ticker_x)
        if verbose:
            print(f"\n  📊 Chart saved: Research/results/regime_eval_{ticker_y}_{ticker_x}.png")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Chart save failed: {e}")

    return evaluation


def _save_regime_comparison_chart(econ_df: pd.DataFrame, ticker_y: str, ticker_x: str):
    """Save bar chart comparing regime detectors by Sharpe."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = econ_df['method'].values
    x = np.arange(len(methods))

    # Sharpe comparison
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    bars = axes[0].bar(x, econ_df['Sharpe'].values, color=colors, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel('OOS Sharpe Ratio')
    axes[0].set_title(f'Regime Detector Comparison — Sharpe\n{ticker_y}/{ticker_x}',
                      fontweight='bold')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, econ_df['Sharpe'].values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Return comparison
    bars2 = axes[1].bar(x, econ_df['Total_Return_%'].values, color=colors, edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('OOS Total Return (%)')
    axes[1].set_title(f'Regime Detector Comparison — Return\n{ticker_y}/{ticker_x}',
                      fontweight='bold')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, econ_df['Total_Return_%'].values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:+.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'regime_eval_{ticker_y}_{ticker_x}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import argparse

    parser = argparse.ArgumentParser(
        description='Regime evaluation with optional RF probability gate'
    )
    parser.add_argument('--pair', nargs=2, default=['BAC', 'PNC'],
                        help='Pair tickers, e.g. BAC PNC')
    parser.add_argument('--prob-gate', type=float, default=0.6,
                        help='RF probability gate threshold for NORMAL')
    args = parser.parse_args()

    result = run_regime_evaluation(
        args.pair[0], args.pair[1],
        prob_gate_threshold=args.prob_gate,
        verbose=True,
    )
    print(f"\n  Summary: {result['summary']}")
