import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. ADAPTIVE KALMAN FILTER
# =============================================================================

class ConservativeKalman:
    """
    Adaptive Kalman filter for dynamic hedge ratio estimation.

    The filter adapts its process noise based on the detected market regime,
    allowing faster adaptation during volatile/crisis periods and smoother
    tracking during normal markets.

    Uses Median Absolute Deviation (MAD) for robust outlier detection in
    the innovation sequence, preventing the filter from overreacting to
    extreme price moves.

    Parameters
    ----------
    delta : float
        Initial state covariance.
    var_e : float
        Observation noise variance.
    var_eta : float
        Base process noise variance (scaled by regime).
    """

    def __init__(self, delta: float = 1e-4, var_e: float = 1e-3, var_eta: float = 1e-4):
        self.delta = delta
        self.var_e = var_e
        self.var_eta = var_eta
        self.wt = None          # State estimate (hedge ratio)
        self.Ct = None          # State covariance
        self.At = None          # Predicted state covariance
        self.Rt = None          # Innovation covariance
        self.innovation_history: list = []
        self.max_history: int = 50

    def initialize(self, y0: float, x0: float) -> None:
        self.wt = y0 / x0 if x0 != 0 else 0.0
        self.Ct = self.delta

    def update(self, y: float, x: float, regime: str = 'normal') -> tuple:
        """
        Single-step Kalman filter update.

        Parameters
        ----------
        y : float
            Dependent asset price.
        x : float
            Independent asset price.
        regime : str
            Market regime: 'normal', 'volatile', or 'crisis'.

        Returns
        -------
        tuple
            (innovation, sqrt_innovation_variance)
        """
        if self.wt is None:
            self.initialize(y, x)
            return 0.0, np.sqrt(self.var_e)

        # Regime-dependent process noise
        noise_multiplier = {'crisis': 5.0, 'volatile': 2.0, 'normal': 1.0}
        var_eta_adjusted = self.var_eta * noise_multiplier.get(regime, 1.0)

        # Prediction step
        wt_minus = self.wt
        self.At = self.Ct + var_eta_adjusted

        # Innovation
        et = y - wt_minus * x
        self.Rt = self.At * x**2 + self.var_e

        # Track innovation history for MAD outlier detection
        self.innovation_history.append(et)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        # MAD-based outlier detection
        if len(self.innovation_history) >= 10:
            innovations = np.array(self.innovation_history)
            median = np.median(innovations)
            mad = np.median(np.abs(innovations - median))

            if mad > 1e-6 and abs(et - median) > 3.0 * mad:
                # Outlier: reduce Kalman gain to avoid overreaction
                Kt = 0.05 * (self.At * x / self.Rt if self.Rt != 0 else 0)
            else:
                Kt = self.At * x / self.Rt if self.Rt != 0 else 0
        else:
            Kt = self.At * x / self.Rt if self.Rt != 0 else 0

        # State update
        self.wt = wt_minus + Kt * et
        self.Ct = self.At - Kt * x * self.At

        return et, np.sqrt(self.Rt)

    def get_hedge_ratio(self) -> float:
        return self.wt if self.wt is not None else 0.0


# =============================================================================
# 2. REGIME CLASSIFIER
# =============================================================================

class StrictRegimeClassifier:
    """
    3-tier market regime classifier using Random Forest.

    Classifies each trading day into:
        - NORMAL (2): Safe to trade with standard parameters
        - VOLATILE (1): Trade with tighter parameters
        - CRISIS (0): Block new entries

    Features include market volatility at multiple horizons, drawdowns,
    returns momentum, trend indicators, and pair-specific metrics.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 6,
                 random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=40,
            min_samples_leaf=20,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names: list = []

    def create_features(self, prices: pd.DataFrame,
                        market_index: pd.Series) -> pd.DataFrame:
        """Build feature matrix from price data and market index."""
        features = pd.DataFrame(index=prices.index)
        market_returns = market_index.pct_change()

        # Volatility at multiple horizons
        for window in [5, 10, 20, 40, 60]:
            features[f'vol_{window}'] = (
                market_returns.rolling(window).std() * np.sqrt(252)
            )

        # Volatility trend and acceleration
        features['vol_trend'] = features['vol_5'] - features['vol_60']
        features['vol_accel'] = features['vol_5'] / (features['vol_20'] + 1e-8)

        # Drawdowns at multiple horizons
        for window in [10, 20, 40, 60]:
            rolling_max = market_index.rolling(window, min_periods=1).max()
            features[f'dd_{window}'] = (market_index - rolling_max) / rolling_max

        # Returns at multiple horizons
        for window in [5, 10, 20]:
            features[f'ret_{window}'] = market_index.pct_change(window)

        # Trend indicators
        sma_20 = market_index.rolling(20).mean()
        sma_50 = market_index.rolling(50).mean()
        features['trend'] = (market_index - sma_20) / sma_20
        features['trend_strength'] = (sma_20 - sma_50) / sma_50

        # Range expansion
        hl_range = market_index.rolling(20).max() - market_index.rolling(20).min()
        features['range_expansion'] = hl_range / market_index.rolling(60).mean()

        # Pair-specific features
        if len(prices.columns) >= 2:
            rolling_corr = prices.iloc[:, 0].rolling(30).corr(prices.iloc[:, 1])
            features['pair_corr'] = rolling_corr
            features['corr_change'] = rolling_corr.diff(10)

            pair_ratio = prices.iloc[:, 0] / prices.iloc[:, 1]
            features['pair_vol'] = (
                pair_ratio.pct_change().rolling(20).std() * np.sqrt(252)
            )
            features['pair_vol_spike'] = (
                features['pair_vol'] / features['pair_vol'].rolling(60).mean()
            )

        self.feature_names = list(features.columns)
        return features.ffill().fillna(0)

    def label_regimes(self, features: pd.DataFrame,
                      vol_crisis: float = 0.35, vol_volatile: float = 0.22,
                      dd_crisis: float = -0.20, dd_volatile: float = -0.10
                      ) -> pd.Series:
        """Generate regime labels from features using threshold rules."""
        labels = pd.Series(2, index=features.index)  # Default: NORMAL

        volatile_cond = (
            (features['vol_20'] > vol_volatile) |
            (features['dd_60'] < dd_volatile) |
            (features.get('vol_accel', pd.Series(1, index=features.index)) > 1.8) |
            (features.get('pair_vol_spike', pd.Series(1, index=features.index)) > 1.5)
        )
        labels[volatile_cond] = 1

        crisis_cond = (
            (features['vol_20'] > vol_crisis) |
            (features['dd_60'] < dd_crisis) |
            (features.get('vol_accel', pd.Series(1, index=features.index)) > 2.5) |
            (features.get('pair_corr', pd.Series(1, index=features.index)) < -0.1)
        )
        labels[crisis_cond] = 0

        return labels

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train the RF classifier. Call ONCE on training data only."""
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        logger.info(
            f"Regime classifier trained on {len(X)} samples: "
            f"NORMAL={int((y == 2).sum())}, VOLATILE={int((y == 1).sum())}, "
            f"CRISIS={int((y == 0).sum())}"
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regimes. Classifier must be trained first."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        X_scaled = self.scaler.transform(features.fillna(0))
        return pd.Series(self.model.predict(X_scaled), index=features.index)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get class probabilities for downstream analysis."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        X_scaled = self.scaler.transform(features.fillna(0))
        return self.model.predict_proba(X_scaled)


# =============================================================================
# 3. Z-SCORE AND SIGNAL GENERATION
# =============================================================================

def calculate_zscore(spread: pd.Series, window: int = 40) -> pd.Series:
    """
    Calculate rolling z-score of the spread.
    Uses expanding window for the first `window` observations.
    """
    if len(spread) < window:
        mean = spread.expanding(min_periods=10).mean()
        std = spread.expanding(min_periods=10).std()
    else:
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()

    return (spread - mean) / (std + 1e-8)


def calculate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life via OLS on spread lag."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    spread_diff = spread_diff.iloc[1:]
    spread_lag = spread_lag.iloc[1:]

    if len(spread_lag) < 20:
        return 20.0

    model = LinearRegression()
    model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
    theta = model.coef_[0]
    half_life = -np.log(2) / theta if theta < 0 else np.inf
    return float(np.clip(half_life, 5, 60))


def generate_signals(z_scores: pd.Series, regimes: pd.Series,
                     entry_z_normal: float = 1.5, exit_z_normal: float = 0.5,
                     entry_z_volatile: float = 2.0, exit_z_volatile: float = 0.3,
                     min_hold_days: int = 3, max_hold_days: int = 30,
                     stop_loss_mult: float = 2.0) -> pd.Series:
    """
    Generate trading signals with regime gating.

    Rules:
        - CRISIS (0): Block new entries; open positions held with wide stop-loss.
        - VOLATILE (1): Higher entry threshold, tighter exit.
        - NORMAL (2): Standard thresholds.
    """
    signals = pd.Series(0, index=z_scores.index)
    position = 0
    entry_idx = None
    entry_z_val = None

    for i in range(len(z_scores)):
        z = z_scores.iloc[i]
        regime = regimes.iloc[i]

        if pd.isna(z) or pd.isna(regime):
            signals.iloc[i] = position
            continue

        # CRISIS: block new entries, hold open positions with wide stop
        if regime == 0:
            if position != 0:
                days_held = i - entry_idx if entry_idx is not None else 0
                stop = (
                    (position == 1 and z < -(entry_z_normal * stop_loss_mult)) or
                    (position == -1 and z > (entry_z_normal * stop_loss_mult)) or
                    days_held > max_hold_days
                )
                if stop:
                    position, entry_idx, entry_z_val = 0, None, None
            signals.iloc[i] = position
            continue

        # Regime-dependent thresholds
        if regime == 1:  # VOLATILE
            entry_thresh = entry_z_volatile
            exit_thresh = exit_z_volatile
        else:            # NORMAL
            entry_thresh = entry_z_normal
            exit_thresh = exit_z_normal

        # Entry logic
        if position == 0:
            if z > entry_thresh:
                position, entry_idx, entry_z_val = -1, i, z
            elif z < -entry_thresh:
                position, entry_idx, entry_z_val = 1, i, z

        # Exit logic
        elif position != 0:
            days_held = i - entry_idx if entry_idx is not None else 0

            exit_conds = [
                abs(z) < exit_thresh,
                (position == 1 and z < -entry_thresh * stop_loss_mult),
                (position == -1 and z > entry_thresh * stop_loss_mult),
                days_held > max_hold_days,
            ]

            if days_held >= min_hold_days and any(exit_conds):
                position, entry_idx, entry_z_val = 0, None, None
            elif days_held < min_hold_days:
                stop = (
                    (position == 1 and z < -entry_thresh * stop_loss_mult) or
                    (position == -1 and z > entry_thresh * stop_loss_mult) or
                    days_held > max_hold_days
                )
                if stop:
                    position, entry_idx, entry_z_val = 0, None, None

        signals.iloc[i] = position

    return signals


# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================

class ConservativeSystem:
    """
    Full backtest engine with strict temporal split enforcement.

    The system enforces the following protocol:
        1. Features/labels are computed over the full data range.
        2. The RF regime classifier is trained ONLY on data up to train_end_date.
        3. The trained classifier is frozen and applied to the remaining data.
        4. Transaction costs are integrated inside the backtest loop.
        5. Returns are computed over the full range; the caller slices by date
           to extract train/validate/test performance separately.
    """

    def __init__(self):
        self.kalman = ConservativeKalman()
        self.regime_classifier = StrictRegimeClassifier()
        self.results = None
        self.train_end_date = None

    def run_backtest(self, stock_y: pd.Series, stock_x: pd.Series,
                     market_index: pd.Series,
                     train_end_date: str = '2020-12-31',
                     entry_z_normal: float = 1.5, exit_z_normal: float = 0.5,
                     entry_z_volatile: float = 2.0, exit_z_volatile: float = 0.3,
                     min_hold_days: int = 3, max_hold_days: int = 30,
                     stop_loss_mult: float = 2.0,
                     z_score_window: int = 40,
                     slippage_bps: float = 5.0,
                     commission_per_trade: float = 1.0,
                     notional_per_leg: float = 50000.0,
                     market_impact: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
        """
        Run a full backtest with strict temporal regime training.

        Parameters
        ----------
        stock_y, stock_x : pd.Series
            Price series for the two assets.
        market_index : pd.Series
            Market benchmark (e.g., SPY).
        train_end_date : str
            Last date (inclusive) of the training window for the RF classifier.
            The classifier sees NO data after this date during training.
        slippage_bps : float
            Round-trip slippage in basis points, deducted per trade.
        commission_per_trade : float
            Fixed commission in dollars per trade (each signal change = 2 legs).
        notional_per_leg : float
            Assumed notional value per leg for commission/impact calculation.
        market_impact : bool
            If True, add Almgren-style sqrt market impact (proportional to
            sqrt(notional / ADV)). Typically adds 1-3 bps.
        """
        # Align data
        data = pd.DataFrame({
            'Y': stock_y, 'X': stock_x, 'Market': market_index
        }).dropna()

        if verbose:
            print(f"  Backtest: {data.index[0].date()} to {data.index[-1].date()} "
                  f"({len(data)} days)")

        self.train_end_date = pd.Timestamp(train_end_date)
        train_mask = data.index <= self.train_end_date

        # --- Step 1: Compute features & labels over full range ---
        prices_df = pd.DataFrame({'Y': data['Y'], 'X': data['X']})
        features = self.regime_classifier.create_features(prices_df, data['Market'])
        labels = self.regime_classifier.label_regimes(features)

        # --- Step 2: Train RF ONLY on training data ---
        train_features = features[train_mask]
        train_labels = labels[train_mask]

        if len(train_features) < 100:
            raise ValueError(
                f"Insufficient training data: {len(train_features)} days. "
                f"Need >= 100. Check train_end_date={train_end_date}."
            )

        self.regime_classifier.train(train_features, train_labels)

        # --- Step 3: Predict regimes with FROZEN classifier ---
        data['regime'] = self.regime_classifier.predict(features)

        if verbose:
            train_days = int(train_mask.sum())
            oos_days = len(data) - train_days
            rc = data['regime'].value_counts().to_dict()
            print(f"  Train: {train_days}d | OOS: {oos_days}d | "
                  f"NORMAL:{rc.get(2,0)} VOLATILE:{rc.get(1,0)} CRISIS:{rc.get(0,0)}")

        # --- Step 4: Run Kalman filter ---
        regime_map = {0: 'crisis', 1: 'volatile', 2: 'normal'}
        spreads, spread_stds, hedge_ratios = [], [], []

        for i in range(len(data)):
            y_val = data['Y'].iloc[i]
            x_val = data['X'].iloc[i]
            regime_str = regime_map.get(int(data['regime'].iloc[i]), 'normal')
            et, sqrt_Qt = self.kalman.update(y_val, x_val, regime=regime_str)
            spreads.append(et)
            spread_stds.append(sqrt_Qt)
            hedge_ratios.append(self.kalman.get_hedge_ratio())

        data['spread'] = spreads
        data['spread_std'] = spread_stds
        data['hedge_ratio'] = hedge_ratios

        # --- Step 5: Z-scores ---
        data['z_score'] = calculate_zscore(
            pd.Series(spreads, index=data.index), window=z_score_window
        )

        # --- Step 6: Generate signals ---
        data['final_signal'] = generate_signals(
            data['z_score'], data['regime'],
            entry_z_normal=entry_z_normal, exit_z_normal=exit_z_normal,
            entry_z_volatile=entry_z_volatile, exit_z_volatile=exit_z_volatile,
            min_hold_days=min_hold_days, max_hold_days=max_hold_days,
            stop_loss_mult=stop_loss_mult
        )

        # --- Step 7: Returns WITH integrated transaction costs ---
        data['returns_Y'] = data['Y'].pct_change()
        data['returns_X'] = data['X'].pct_change()
        data['spread_return'] = data['returns_Y'] - data['hedge_ratio'] * data['returns_X']

        # Gross return
        data['strategy_return_gross'] = data['final_signal'].shift(1) * data['spread_return']

        # --- Transaction Cost Model ---
        # Component 1: Slippage (proportional to signal change magnitude)
        signal_changes = data['final_signal'].diff().abs().fillna(0)
        slippage_cost = signal_changes * (slippage_bps / 10_000)

        # Component 2: Commission ($1 per trade, 2 legs per signal change)
        # Convert to return-equivalent: commission / notional
        trades_occurred = (signal_changes > 0).astype(float)
        commission_cost = trades_occurred * (commission_per_trade * 2) / notional_per_leg

        # Component 3: Almgren square-root market impact
        # Impact ≈ σ_daily × sqrt(notional / ADV)
        # Simplified: estimate ADV from rolling volume proxy (price × typical shares)
        if market_impact:
            daily_vol_y = data['returns_Y'].rolling(20).std().fillna(data['returns_Y'].std())
            # Conservative estimate: impact ≈ 0.1 × daily_vol × sqrt(participation_rate)
            # With notional=50k and avg stock ADV ≈ $200M, participation ≈ 0.025%
            # impact ≈ 0.1 × σ × sqrt(0.00025) ≈ 0.0016 × σ
            impact_per_trade = 0.1 * daily_vol_y * np.sqrt(notional_per_leg / 200_000_000)
            market_impact_cost = trades_occurred * impact_per_trade
        else:
            market_impact_cost = 0.0

        # Total transaction cost
        data['cost_slippage'] = slippage_cost
        data['cost_commission'] = commission_cost
        data['cost_impact'] = market_impact_cost if isinstance(market_impact_cost, pd.Series) \
            else pd.Series(0.0, index=data.index)
        data['transaction_cost'] = slippage_cost + commission_cost + data['cost_impact']

        # Net return
        data['strategy_return'] = data['strategy_return_gross'] - data['transaction_cost']

        # Baseline: no regime filter (same cost model)
        baseline_sig = generate_signals(
            data['z_score'], pd.Series(2, index=data.index),
            entry_z_normal=entry_z_normal, exit_z_normal=exit_z_normal,
            min_hold_days=min_hold_days, max_hold_days=max_hold_days,
            stop_loss_mult=stop_loss_mult
        )
        b_changes = baseline_sig.diff().abs().fillna(0)
        b_trades = (b_changes > 0).astype(float)
        baseline_cost = (b_changes * (slippage_bps / 10_000)
                         + b_trades * (commission_per_trade * 2) / notional_per_leg
                         + b_trades * data['cost_impact'] / trades_occurred.replace(0, 1))
        baseline_cost = baseline_cost.fillna(0)
        data['baseline_return'] = baseline_sig.shift(1) * data['spread_return'] - baseline_cost

        # Cumulative
        data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
        data['baseline_cumulative'] = (1 + data['baseline_return']).cumprod()
        data['market_return'] = data['Market'].pct_change()
        data['market_cumulative'] = (1 + data['market_return']).cumprod()

        # Period marker
        data['period'] = 'train'
        data.loc[data.index > self.train_end_date, 'period'] = 'oos'

        self.results = data
        return data

    def get_performance_metrics(self, period: str = 'all') -> dict:
        """Compute performance metrics. period: 'train', 'oos', or 'all'."""
        if self.results is None:
            raise ValueError("Must run backtest first")
        data = self.results
        if period == 'train':
            data = data[data['period'] == 'train']
        elif period in ('oos', 'test', 'validate'):
            data = data[data['period'] == 'oos']

        s_ret = data['strategy_return'].dropna()
        b_ret = data['baseline_return'].dropna()
        m_ret = data['market_return'].dropna()

        metrics = {
            'Strategy (Net)': self._calc(s_ret),
            'No-Regime Baseline': self._calc(b_ret),
            'Market (SPY)': self._calc(m_ret),
        }

        # Gross vs Net breakdown
        s_gross = data['strategy_return_gross'].dropna()
        total_cost = data['transaction_cost'].sum()
        cost_slippage = data['cost_slippage'].sum() if 'cost_slippage' in data else 0
        cost_commission = data['cost_commission'].sum() if 'cost_commission' in data else 0
        cost_impact = data['cost_impact'].sum() if 'cost_impact' in data else 0
        metrics['Strategy (Gross)'] = self._calc(s_gross)
        metrics['Cost Breakdown'] = {
            'Total Cost (bps equiv)': f"{total_cost * 10000:.1f}",
            'Slippage': f"{cost_slippage * 10000:.1f} bps",
            'Commission': f"{cost_commission * 10000:.1f} bps",
            'Market Impact': f"{cost_impact * 10000:.1f} bps",
        }

        total = len(data)
        rc = data['regime'].value_counts()
        metrics['Regime Info'] = {
            'Total Days': total,
            'Normal': int(rc.get(2, 0)),
            'Volatile': int(rc.get(1, 0)),
            'Crisis': int(rc.get(0, 0)),
            'Crisis %': f"{rc.get(0, 0) / max(total, 1) * 100:.1f}%",
        }
        return metrics

    def get_trade_analysis(self, period: str = 'all') -> pd.DataFrame:
        """Extract individual trade records."""
        if self.results is None:
            raise ValueError("Must run backtest first")
        data = self.results.copy()
        if period == 'train':
            data = data[data['period'] == 'train']
        elif period in ('oos', 'test', 'validate'):
            data = data[data['period'] == 'oos']

        data['position_change'] = data['final_signal'].diff()
        trades, current = [], None

        for i in range(len(data)):
            chg = data['position_change'].iloc[i]
            sig = data['final_signal'].iloc[i]
            if chg != 0 and sig != 0 and current is None:
                current = {
                    'entry_date': data.index[i],
                    'entry_signal': sig,
                    'entry_z': data['z_score'].iloc[i],
                    'entry_hedge': data['hedge_ratio'].iloc[i],
                    'entry_regime': int(data['regime'].iloc[i]),
                }
            elif chg != 0 and sig == 0 and current is not None:
                eloc = data.index.get_loc(current['entry_date'])
                pnl = data['strategy_return'].iloc[eloc:i + 1].sum()
                current.update({
                    'exit_date': data.index[i],
                    'exit_z': data['z_score'].iloc[i],
                    'exit_regime': int(data['regime'].iloc[i]),
                    'duration_days': (data.index[i] - current['entry_date']).days,
                    'pnl': pnl,
                    'pnl_pct': pnl * 100,
                    'profitable': pnl > 0,
                })
                trades.append(current)
                current = None
        return pd.DataFrame(trades)

    def slice_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return results sliced to a specific date range."""
        if self.results is None:
            raise ValueError("Must run backtest first")
        mask = (self.results.index >= start_date) & (self.results.index <= end_date)
        return self.results[mask]

    @staticmethod
    def _calc(returns: pd.Series) -> dict:
        if len(returns) == 0 or returns.std() == 0:
            return {'Total Return': '0.00%', 'Annual Return': '0.00%',
                    'Annual Vol': '0.00%', 'Sharpe': '0.000',
                    'Max DD': '0.00%', 'Win Rate': '0.0%', 'Days': 0}
        tr = (1 + returns).prod() - 1
        n = len(returns)
        ar = (1 + tr) ** (252 / n) - 1
        av = returns.std() * np.sqrt(252)
        sh = ar / av if av != 0 else 0
        cum = (1 + returns).cumprod()
        dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
        wr = (returns > 0).sum() / n
        return {
            'Total Return': f"{tr * 100:.2f}%",
            'Annual Return': f"{ar * 100:.2f}%",
            'Annual Vol': f"{av * 100:.2f}%",
            'Sharpe': f"{sh:.3f}",
            'Max DD': f"{dd * 100:.2f}%",
            'Win Rate': f"{wr * 100:.1f}%",
            'Days': n,
        }


# =============================================================================
# 5. DATA DOWNLOAD
# =============================================================================

def download_data(ticker_y: str, ticker_x: str, market_ticker: str = 'SPY',
                  start_date: str = '2015-01-01',
                  end_date: str = '2025-06-30') -> tuple:
    """Download price data for a pair and market benchmark."""
    data_y = yf.download(ticker_y, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
    data_x = yf.download(ticker_x, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
    data_m = yf.download(market_ticker, start=start_date, end=end_date,
                         progress=False, auto_adjust=True)

    def extract_close(df):
        if df.empty:
            raise ValueError("Downloaded data is empty")
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                return df['Close'].iloc[:, 0].squeeze()
            return df.iloc[:, 0].squeeze()
        if 'Close' in df.columns:
            return df['Close'].squeeze()
        return df.iloc[:, 0].squeeze()

    return pd.Series(extract_close(data_y)), pd.Series(extract_close(data_x)), pd.Series(extract_close(data_m))


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("=" * 70)
    print("CONSERVATIVE PAIRS TRADING — Academic Backtest Protocol")
    print("=" * 70)

    bac, pnc, spy = download_data('BAC', 'PNC', 'SPY', '2015-01-01', '2025-06-30')
    system = ConservativeSystem()
    results = system.run_backtest(
        bac, pnc, spy, train_end_date='2020-12-31',
        entry_z_normal=1.5, exit_z_normal=0.5,
        entry_z_volatile=2.0, exit_z_volatile=0.3,
        min_hold_days=3, max_hold_days=30, z_score_window=40, slippage_bps=5.0,
    )

    for pn in ['train', 'oos']:
        print(f"\n{'=' * 50}\n  {pn.upper()} PERIOD\n{'=' * 50}")
        for name, m in system.get_performance_metrics(period=pn).items():
            if name == 'Regime Info':
                continue
            print(f"\n  {name}:")
            for k, v in m.items():
                print(f"    {k:.<25} {v}")

    trades = system.get_trade_analysis(period='oos')
    if len(trades) > 0:
        print(f"\n  OOS Trades: {len(trades)} | "
              f"Win Rate: {trades['profitable'].mean()*100:.1f}% | "
              f"Avg PnL: {trades['pnl_pct'].mean():.2f}%")
