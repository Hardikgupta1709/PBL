"""
Feature Importance Analysis for RF Regime Classifier
======================================================
Answers: "Which features actually drive regime detection?"

Methods:
    1. SHAP TreeExplainer — exact Shapley values for tree ensembles
    2. Scikit-learn built-in feature importance (Gini/MDI)
    3. Permutation importance with 10 repeats (model-agnostic)
    4. Feature ablation — remove one feature group, measure accuracy drop

Output:
    - Feature ranking table (CSV)
    - SHAP summary data (CSV)
    - Feature group ablation results (CSV)

Reference:
    Lundberg & Lee (2017) "A Unified Approach to Interpreting Model
    Predictions" — SHAP values for tree models
"""

import numpy as np
import pandas as pd
import os
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import (
    StrictRegimeClassifier, download_data
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE GROUPS (for group ablation)
# =============================================================================

FEATURE_GROUPS = {
    'Volatility': ['vol_5', 'vol_10', 'vol_20', 'vol_40', 'vol_60'],
    'Vol Dynamics': ['vol_trend', 'vol_accel'],
    'Drawdowns': ['dd_10', 'dd_20', 'dd_40', 'dd_60'],
    'Returns': ['ret_5', 'ret_10', 'ret_20'],
    'Trend': ['trend', 'trend_strength'],
    'Range': ['range_expansion'],
    'Pair Corr': ['pair_corr', 'corr_change'],
    'Pair Vol': ['pair_vol', 'pair_vol_spike'],
}


# =============================================================================
# 1. TRAIN CLASSIFIER AND GET FEATURES
# =============================================================================

def prepare_classifier_data(ticker_y: str, ticker_x: str,
                            start_date: str = '2015-01-01',
                            end_date: str = '2025-06-30',
                            train_end_date: str = '2020-12-31'):
    """
    Download data, create features, train classifier on training period.
    Returns classifier, features, labels, train/test masks.
    """
    stock_y, stock_x, market = download_data(
        ticker_y, ticker_x, 'SPY', start_date, end_date
    )
    data = pd.DataFrame({
        'Y': stock_y, 'X': stock_x, 'Market': market
    }).dropna()

    train_end = pd.Timestamp(train_end_date)
    train_mask = data.index <= train_end
    test_mask = ~train_mask

    classifier = StrictRegimeClassifier()
    prices_df = pd.DataFrame({'Y': data['Y'], 'X': data['X']})
    features = classifier.create_features(prices_df, data['Market'])
    labels = classifier.label_regimes(features)

    # Remove NaN rows
    valid = ~(features.isna().any(axis=1) | labels.isna())
    features = features[valid]
    labels = labels[valid]
    train_mask = train_mask[valid]
    test_mask = test_mask[valid]

    # Train on training data only
    classifier.train(features[train_mask], labels[train_mask])

    return classifier, features, labels, train_mask, test_mask


# =============================================================================
# 2. SHAP ANALYSIS
# =============================================================================

def shap_analysis(classifier: StrictRegimeClassifier,
                  features: pd.DataFrame,
                  test_mask: pd.Series,
                  max_samples: int = 500,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Compute SHAP values for the RF regime classifier.

    Uses TreeExplainer for exact Shapley values on tree ensembles.
    Limited to max_samples from the test set for computational efficiency.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  1. SHAP FEATURE IMPORTANCE")
        print(f"{'=' * 70}")

    try:
        import shap
    except ImportError:
        print("  ⚠️ SHAP not installed. Run: pip install shap")
        return pd.DataFrame()

    # Use test data for SHAP
    X_test = features[test_mask]
    X_test_scaled = classifier.scaler.transform(X_test.fillna(0))

    # Subsample for speed
    n_samples = min(max_samples, len(X_test_scaled))
    idx = np.random.RandomState(42).choice(len(X_test_scaled), n_samples, replace=False)
    X_sample = X_test_scaled[idx]

    if verbose:
        print(f"  Computing SHAP values on {n_samples} test samples...")

    # TreeExplainer — exact for tree models
    explainer = shap.TreeExplainer(classifier.model)
    shap_values = explainer.shap_values(X_sample)

    # SHAP 0.51+ returns ndarray (samples, features, classes) for multi-class
    # Older versions return list of arrays [class0_arr, class1_arr, ...]
    feature_names = list(features.columns)
    class_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))  # avg over samples & classes
        per_class_abs = {}
        for cls_idx in range(shap_values.shape[2]):
            cls_name = class_names.get(cls_idx, f'Class_{cls_idx}')
            per_class_abs[cls_name] = np.abs(shap_values[:, :, cls_idx]).mean(axis=0)
    elif isinstance(shap_values, list):
        mean_abs_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
        per_class_abs = {}
        for cls_idx, sv in enumerate(shap_values):
            cls_name = class_names.get(cls_idx, f'Class_{cls_idx}')
            per_class_abs[cls_name] = np.abs(sv).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        per_class_abs = {}

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    shap_df['Rank'] = range(1, len(shap_df) + 1)

    # Per-class SHAP columns
    for cls_name, abs_mean in per_class_abs.items():
        shap_df[f'SHAP_{cls_name}'] = [
            abs_mean[feature_names.index(f)] for f in shap_df['Feature']
        ]

    if verbose:
        print(f"\n  Top 10 Features by Mean |SHAP|:")
        for _, row in shap_df.head(10).iterrows():
            bar = "█" * max(1, int(row['Mean_Abs_SHAP'] * 500))
            print(f"    {int(row['Rank']):>2}. {row['Feature']:.<25} "
                  f"{row['Mean_Abs_SHAP']:.4f}  {bar}")

    return shap_df


# =============================================================================
# 3. SCIKIT-LEARN BUILT-IN IMPORTANCE (MDI / Gini)
# =============================================================================

def builtin_importance(classifier: StrictRegimeClassifier,
                       features: pd.DataFrame,
                       verbose: bool = True) -> pd.DataFrame:
    """Extract Gini / MDI importance from the Random Forest."""
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  2. BUILT-IN (GINI/MDI) IMPORTANCE")
        print(f"{'=' * 70}")

    importances = classifier.model.feature_importances_
    feature_names = list(features.columns)

    df = pd.DataFrame({
        'Feature': feature_names,
        'Gini_Importance': importances,
    }).sort_values('Gini_Importance', ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    if verbose:
        for _, row in df.head(10).iterrows():
            bar = "█" * max(1, int(row['Gini_Importance'] * 100))
            print(f"    {int(row['Rank']):>2}. {row['Feature']:.<25} "
                  f"{row['Gini_Importance']:.4f}  {bar}")

    return df


# =============================================================================
# 4. PERMUTATION IMPORTANCE
# =============================================================================

def permutation_importance_analysis(classifier: StrictRegimeClassifier,
                                    features: pd.DataFrame,
                                    labels: pd.Series,
                                    test_mask: pd.Series,
                                    n_repeats: int = 10,
                                    verbose: bool = True) -> pd.DataFrame:
    """
    Model-agnostic permutation importance.

    Shuffles each feature one at a time and measures accuracy drop.
    Uses 10 repeats for stability.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  3. PERMUTATION IMPORTANCE ({n_repeats} repeats)")
        print(f"{'=' * 70}")

    X_test = features[test_mask].fillna(0)
    y_test = labels[test_mask]
    X_test_scaled = classifier.scaler.transform(X_test)

    result = permutation_importance(
        classifier.model, X_test_scaled, y_test,
        n_repeats=n_repeats, random_state=42,
        scoring='accuracy',
    )

    feature_names = list(features.columns)
    df = pd.DataFrame({
        'Feature': feature_names,
        'Perm_Importance_Mean': result.importances_mean,
        'Perm_Importance_Std': result.importances_std,
    }).sort_values('Perm_Importance_Mean', ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    if verbose:
        for _, row in df.head(10).iterrows():
            bar = "█" * max(1, int(row['Perm_Importance_Mean'] * 500))
            print(f"    {int(row['Rank']):>2}. {row['Feature']:.<25} "
                  f"{row['Perm_Importance_Mean']:.4f} ± {row['Perm_Importance_Std']:.4f}  {bar}")

    return df


# =============================================================================
# 5. FEATURE GROUP ABLATION
# =============================================================================

def feature_group_ablation(classifier_factory,
                           features: pd.DataFrame,
                           labels: pd.Series,
                           train_mask: pd.Series,
                           test_mask: pd.Series,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Remove one feature group at a time and measure accuracy change.

    This is the most rigorous form of importance — it captures feature
    interactions that SHAP and permutation importance can miss.

    Parameters
    ----------
    classifier_factory : callable
        Function that returns a fresh StrictRegimeClassifier instance.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  4. FEATURE GROUP ABLATION")
        print(f"{'=' * 70}")

    X_train = features[train_mask].fillna(0)
    y_train = labels[train_mask]
    X_test = features[test_mask].fillna(0)
    y_test = labels[test_mask]

    # Baseline: all features
    base_clf = classifier_factory()
    base_clf.train(X_train, y_train)
    base_preds = base_clf.predict(X_test)
    base_acc = accuracy_score(y_test, base_preds)
    base_f1 = f1_score(y_test, base_preds, average='weighted')

    if verbose:
        print(f"  Baseline (all features): Acc={base_acc:.4f} F1={base_f1:.4f}")

    rows = [{'Group': 'All Features', 'Accuracy': base_acc, 'F1': base_f1,
             'Acc_Delta': 0.0, 'F1_Delta': 0.0, 'Features_Removed': 0}]

    for group_name, group_features in FEATURE_GROUPS.items():
        # Find features that exist in our feature set
        existing = [f for f in group_features if f in features.columns]
        if len(existing) == 0:
            continue

        # Create feature set with group removed
        remaining = [f for f in features.columns if f not in existing]
        X_train_ablated = X_train[remaining]
        X_test_ablated = X_test[remaining]

        try:
            abl_clf = classifier_factory()
            abl_clf.train(X_train_ablated, y_train)
            abl_preds = abl_clf.predict(X_test_ablated)
            abl_acc = accuracy_score(y_test, abl_preds)
            abl_f1 = f1_score(y_test, abl_preds, average='weighted')

            rows.append({
                'Group': f'−{group_name}',
                'Accuracy': abl_acc,
                'F1': abl_f1,
                'Acc_Delta': round(abl_acc - base_acc, 4),
                'F1_Delta': round(abl_f1 - base_f1, 4),
                'Features_Removed': len(existing),
            })

            if verbose:
                icon = "🔴" if (abl_acc - base_acc) < -0.01 else "⚪"
                print(f"  {icon} {f'−{group_name}':.<30} Acc={abl_acc:.4f} "
                      f"(Δ={abl_acc - base_acc:+.4f})  "
                      f"F1={abl_f1:.4f} (Δ={abl_f1 - base_f1:+.4f})  "
                      f"[{len(existing)} features]")

        except Exception as e:
            logger.warning(f"Group ablation '{group_name}' failed: {e}")

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# 6. UNIFIED ANALYSIS RUNNER
# =============================================================================

def run_feature_analysis(ticker_y: str = 'BAC', ticker_x: str = 'PNC',
                         start_date: str = '2015-01-01',
                         end_date: str = '2025-06-30',
                         train_end_date: str = '2020-12-31',
                         verbose: bool = True) -> dict:
    """Run all feature importance analyses for a single pair."""
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  FEATURE IMPORTANCE ANALYSIS: {ticker_y}/{ticker_x}")
        print(f"{'=' * 70}")

    # Prepare data
    classifier, features, labels, train_mask, test_mask = prepare_classifier_data(
        ticker_y, ticker_x, start_date, end_date, train_end_date
    )

    results = {}

    # 1. SHAP
    results['shap'] = shap_analysis(
        classifier, features, test_mask, verbose=verbose
    )

    # 2. Built-in importance
    results['gini'] = builtin_importance(
        classifier, features, verbose=verbose
    )

    # 3. Permutation importance
    results['permutation'] = permutation_importance_analysis(
        classifier, features, labels, test_mask, verbose=verbose
    )

    # 4. Feature group ablation
    def make_classifier():
        return StrictRegimeClassifier()

    results['group_ablation'] = feature_group_ablation(
        make_classifier, features, labels, train_mask, test_mask, verbose=verbose
    )

    # 5. Consensus ranking — merge all three methods
    if len(results['shap']) > 0 and len(results['gini']) > 0 and len(results['permutation']) > 0:
        consensus = results['shap'][['Feature', 'Mean_Abs_SHAP']].merge(
            results['gini'][['Feature', 'Gini_Importance']], on='Feature', how='outer'
        ).merge(
            results['permutation'][['Feature', 'Perm_Importance_Mean']], on='Feature', how='outer'
        ).fillna(0)

        # Rank each method (lower rank = more important)
        for col in ['Mean_Abs_SHAP', 'Gini_Importance', 'Perm_Importance_Mean']:
            consensus[f'{col}_rank'] = consensus[col].rank(ascending=False)

        consensus['Avg_Rank'] = consensus[
            ['Mean_Abs_SHAP_rank', 'Gini_Importance_rank', 'Perm_Importance_Mean_rank']
        ].mean(axis=1)
        consensus = consensus.sort_values('Avg_Rank')
        consensus['Consensus_Rank'] = range(1, len(consensus) + 1)

        results['consensus'] = consensus

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  CONSENSUS RANKING (Top 10)")
            print(f"{'=' * 70}")
            for _, row in consensus.head(10).iterrows():
                print(f"    {int(row['Consensus_Rank']):>2}. {row['Feature']:.<25} "
                      f"SHAP_rk={int(row['Mean_Abs_SHAP_rank'])} "
                      f"Gini_rk={int(row['Gini_Importance_rank'])} "
                      f"Perm_rk={int(row['Perm_Importance_Mean_rank'])} "
                      f"Avg={row['Avg_Rank']:.1f}")

    return results


def run_feature_analysis_multi_pair(pairs: list,
                                    start_date: str = '2015-01-01',
                                    end_date: str = '2025-06-30',
                                    train_end_date: str = '2020-12-31') -> pd.DataFrame:
    """Run feature analysis across multiple pairs and aggregate rankings."""
    all_consensus = []

    for i, (ty, tx) in enumerate(pairs):
        print(f"\n{'─' * 60}")
        print(f"  [{i+1}/{len(pairs)}] Feature Analysis: {ty}/{tx}")
        try:
            results = run_feature_analysis(
                ty, tx, start_date, end_date, train_end_date, verbose=False
            )
            if 'consensus' in results:
                cons = results['consensus'][['Feature', 'Avg_Rank']].copy()
                cons['Pair'] = f"{ty}_{tx}"
                all_consensus.append(cons)
                top3 = cons.nsmallest(3, 'Avg_Rank')['Feature'].tolist()
                print(f"    Top 3: {', '.join(top3)}")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    if len(all_consensus) == 0:
        return pd.DataFrame()

    full = pd.concat(all_consensus, ignore_index=True)
    agg = full.groupby('Feature')['Avg_Rank'].agg(['mean', 'median', 'std', 'count'])
    agg = agg.sort_values('mean')
    agg['Global_Rank'] = range(1, len(agg) + 1)

    print(f"\n{'=' * 70}")
    print(f"  GLOBAL FEATURE RANKING ({len(pairs)} pairs)")
    print(f"{'=' * 70}")
    for feat, row in agg.head(10).iterrows():
        bar = "█" * max(1, int((20 - row['mean']) * 2))
        print(f"    {int(row['Global_Rank']):>2}. {feat:.<25} "
              f"avg_rank={row['mean']:.1f} ± {row['std']:.1f}  "
              f"(seen in {int(row['count'])} pairs)  {bar}")

    return agg


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    results = run_feature_analysis('BAC', 'PNC')

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    for name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            path = os.path.join(results_dir, f'feature_{name}_BAC_PNC.csv')
            df.to_csv(path, index=False)
            print(f"  Saved: {path}")
