import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')



class KalmanFilter:

    
    def __init__(self, delta=1e-4, var_e=1e-3, var_eta=1e-4):
        self.delta = delta
        self.var_e = var_e
        self.var_eta = var_eta
        
        # State variables
        self.wt = None
        self.Ct = None
        self.At = None
        self.Rt = None
        
        # Outlier detection
        self.innovation_history = []
        self.max_history = 50
        
    def initialize(self, y0, x0):
        self.wt = y0 / x0 if x0 != 0 else 0
        self.Ct = self.delta
        
    def update(self, y, x, regime='normal'):
        if self.wt is None:
            self.initialize(y, x)
            return 0, np.sqrt(self.var_e)
        
        # Regime-dependent variance (more noise = faster adaptation)
        var_eta_adjusted = {
            'crisis': self.var_eta * 5,    # Adapt fastest in crisis
            'volatile': self.var_eta * 2,  # Moderate adaptation
            'normal': self.var_eta         # Slowest, most stable
        }.get(regime, self.var_eta)
        
        # Prediction step
        wt_minus = self.wt
        self.At = self.Ct + var_eta_adjusted
        
        # Innovation (prediction error)
        et = y - wt_minus * x
        self.Rt = self.At * x**2 + self.var_e
        
        # Track innovation history for outlier detection
        self.innovation_history.append(et)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)
        
        # Robust outlier detection using MAD
        if len(self.innovation_history) >= 10:
            innovations = np.array(self.innovation_history)
            median = np.median(innovations)
            mad = np.median(np.abs(innovations - median))
            
            # If current innovation is outlier (> 3 MAD from median)
            # reduce Kalman gain to avoid overreacting
            if abs(et - median) > 3 * mad and mad > 1e-6:
                Kt = 0.05 * (self.At * x / self.Rt if self.Rt != 0 else 0)
            else:
                Kt = self.At * x / self.Rt if self.Rt != 0 else 0
        else:
            Kt = self.At * x / self.Rt if self.Rt != 0 else 0
        
        # State update
        self.wt = wt_minus + Kt * et
        self.Ct = self.At - Kt * x * self.At
        
        sqrt_Qt = np.sqrt(self.Rt)
        
        return et, sqrt_Qt
    
    def get_hedge_ratio(self):
        return self.wt if self.wt is not None else 0


class RegimeClassifier:
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=40,
            min_samples_leaf=20,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, prices, market_index, vix=None):
        features = pd.DataFrame(index=prices.index)
        
        market_returns = market_index.pct_change()
        
        # Multiple volatility windows
        for window in [5, 10, 20, 40, 60]:
            features[f'vol_{window}'] = market_returns.rolling(window).std() * np.sqrt(252)
        
        # Volatility trend and acceleration
        features['vol_trend'] = features['vol_5'] - features['vol_60']
        features['vol_accel'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        
        # Multiple drawdown windows
        for window in [10, 20, 40, 60]:
            rolling_max = market_index.rolling(window, min_periods=1).max()
            features[f'dd_{window}'] = (market_index - rolling_max) / rolling_max
        
        # Returns at different horizons
        for window in [5, 10, 20]:
            features[f'ret_{window}'] = market_index.pct_change(window)
        
        # Trend indicators
        sma_20 = market_index.rolling(20).mean()
        sma_50 = market_index.rolling(50).mean()
        features['trend'] = (market_index - sma_20) / sma_20
        features['trend_strength'] = (sma_20 - sma_50) / sma_50
        
        # Range expansion
        high_low_range = market_index.rolling(20).max() - market_index.rolling(20).min()
        features['range_expansion'] = high_low_range / market_index.rolling(60).mean()
        
        # Pair-specific features
        if len(prices.columns) >= 2:
            rolling_corr = prices.iloc[:, 0].rolling(30).corr(prices.iloc[:, 1])
            features['pair_corr'] = rolling_corr
            features['corr_change'] = rolling_corr.diff(10)
            
            pair_ratio = prices.iloc[:, 0] / prices.iloc[:, 1]
            features['pair_vol'] = pair_ratio.pct_change().rolling(20).std() * np.sqrt(252)
            features['pair_vol_spike'] = features['pair_vol'] / features['pair_vol'].rolling(60).mean()
        
        return features.ffill().fillna(0)
    
    def label_regimes(self, features, 
                     vol_crisis=0.35, vol_volatile=0.22,
                     dd_crisis=-0.20, dd_volatile=-0.10):
        labels = pd.Series(2, index=features.index)  # Default: NORMAL
        
        # Mark VOLATILE
        volatile_conditions = (
            (features['vol_20'] > vol_volatile) |
            (features['dd_60'] < dd_volatile) |
            (features.get('vol_accel', 1) > 1.8) |
            (features.get('pair_vol_spike', 1) > 1.5)
        )
        labels[volatile_conditions] = 1
        
        # Mark CRISIS (overrides volatile) — relaxed to avoid over-classification
        crisis_conditions = (
            (features['vol_20'] > vol_crisis) |
            (features['dd_60'] < dd_crisis) |
            (features.get('vol_accel', 1) > 2.5) |
            (features.get('pair_corr', 1) < -0.1)
        )
        labels[crisis_conditions] = 0
        
        return labels
    
    def train(self, features, labels):
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"3-Tier Regime classifier trained on {len(X)} samples")
        print(f"NORMAL: {(y==2).sum()}, VOLATILE: {(y==1).sum()}, CRISIS: {(y==0).sum()}")
        
    def predict(self, features):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(features.fillna(0))
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=features.index)


# PART 3: ADVANCED Z-SCORE CALCULATION

def calculate_robust_zscore(spread, base_window=60):
    # Use expanding window for first base_window days
    if len(spread) < base_window:
        mean = spread.expanding(min_periods=10).mean()
        std = spread.expanding(min_periods=10).std()
    else:
        mean = spread.rolling(window=base_window).mean()
        std = spread.rolling(window=base_window).std()
    
    z_score = (spread - mean) / (std + 1e-8)
    return z_score


def calculate_multi_timeframe_zscore(spread):
    z_30 = calculate_robust_zscore(spread, base_window=30)
    z_60 = calculate_robust_zscore(spread, base_window=60)
    z_90 = calculate_robust_zscore(spread, base_window=90)
    
    return z_30, z_60, z_90

# PART 4: ADVANCED SIGNAL GENERATION

def calculate_position_size(z_score, volatility, base_size=1.0, max_size=1.5):
    z_abs = abs(z_score)
    signal_factor = np.clip((z_abs - 2.0) / 2.0, 0.0, 0.5) + 1.0
    
    vol_factor = np.clip(1.0 - (volatility - 0.15) / 0.15, 0.5, 1.0)
    
    size = base_size * signal_factor * vol_factor
    return np.clip(size, 0.5, max_size)


def calculate_half_life(spread):
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # Align
    spread_diff = spread_diff.iloc[1:]
    spread_lag = spread_lag.iloc[1:]
    
    if len(spread_lag) < 20:
        return 20  # Default
    
    # Regression
    model = LinearRegression()
    model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
    
    theta = model.coef_[0]
    half_life = -np.log(2) / theta if theta < 0 else np.inf
    
    return max(5, min(half_life, 60))  # Clip to [5, 60]


def generate_ultimate_signals(z_scores, regimes, data,
                              entry_z_normal=2.0, exit_z_normal=0.5,
                              entry_z_volatile=2.5, exit_z_volatile=0.3,
                              min_hold_days=3,
                              min_correlation=0.5,
                              use_multi_timeframe=True):
    signals = pd.Series(0, index=z_scores.index)
    position = 0
    entry_date_idx = None
    entry_z = None
    
    # Calculate rolling correlation
    rolling_corr = data['Y'].rolling(30).corr(data['X'])
    
    # Calculate multiple timeframes if enabled
    if use_multi_timeframe:
        z_30, z_60, z_90 = calculate_multi_timeframe_zscore(data['spread'])
    else:
        z_30 = z_60 = z_90 = z_scores
    
    # Calculate half-life for dynamic max hold
    half_life = calculate_half_life(data['spread'].iloc[:min(len(data), 500)])
    max_hold_days = int(half_life * 3)
    
    for i in range(len(z_scores)):
        z = z_scores.iloc[i]
        regime = regimes.iloc[i]
        corr = rolling_corr.iloc[i]
        
        if pd.isna(z) or pd.isna(regime):
            signals.iloc[i] = position
            continue
        
        # CRITICAL: Exit if correlation breaks
        if position != 0 and corr < min_correlation:
            position = 0
            entry_date_idx = None
            entry_z = None
            signals.iloc[i] = position
            continue
        
        # Don't enter if correlation already low
        if position == 0 and corr < min_correlation:
            signals.iloc[i] = position
            continue
        
        # NEVER trade in CRISIS
        if regime == 0:
            position = 0
            entry_date_idx = None
            entry_z = None
            signals.iloc[i] = position
            continue
        
        # Adjust thresholds based on regime
        if regime == 1:  # VOLATILE
            entry_threshold = entry_z_volatile
            exit_threshold = exit_z_volatile
        else:  # NORMAL (regime == 2)
            entry_threshold = entry_z_normal
            exit_threshold = exit_z_normal
        
        # Entry logic with multiple confirmations
        if position == 0:
            if use_multi_timeframe:
                # All three timeframes must agree
                entry_long = (z_30.iloc[i] < -entry_threshold and 
                            z_60.iloc[i] < -entry_threshold and 
                            z_90.iloc[i] < -(entry_threshold * 0.8))
                
                entry_short = (z_30.iloc[i] > entry_threshold and 
                             z_60.iloc[i] > entry_threshold and 
                             z_90.iloc[i] > (entry_threshold * 0.8))
            else:
                entry_long = z < -entry_threshold
                entry_short = z > entry_threshold
            
            # Trend confirmation
            if i >= 5:
                recent_z_trend = z_scores.iloc[i-5:i].mean()
                
                if entry_short and z > recent_z_trend:
                    position = -1
                    entry_date_idx = i
                    entry_z = z
                elif entry_long and z < recent_z_trend:
                    position = 1
                    entry_date_idx = i
                    entry_z = z
            else:
                if entry_short:
                    position = -1
                    entry_date_idx = i
                    entry_z = z
                elif entry_long:
                    position = 1
                    entry_date_idx = i
                    entry_z = z
        
        # Exit logic with multiple conditions
        elif position != 0:
            days_held = i - entry_date_idx if entry_date_idx is not None else 0
            
            # Multiple timeframe exit (any one crosses)
            if use_multi_timeframe:
                exit_condition = (abs(z_30.iloc[i]) < exit_threshold or 
                                abs(z_60.iloc[i]) < exit_threshold)
            else:
                exit_condition = abs(z) < exit_threshold
            
            # Force exit conditions
            exit_conditions = [
                exit_condition,
                
                # Stop-loss: z-score moved 1.5x against us
                (position == 1 and z < -entry_threshold * 1.5),
                (position == -1 and z > entry_threshold * 1.5),
                
                # Regime deteriorated
                (regime == 1 and entry_z is not None and abs(entry_z) < entry_z_volatile),
                
                # Maximum holding period exceeded (dynamic based on half-life)
                days_held > max_hold_days
            ]
            
            # Apply minimum holding period (except for stop-loss)
            if days_held >= min_hold_days:
                if any(exit_conditions):
                    position = 0
                    entry_date_idx = None
                    entry_z = None
            else:
                # Before min holding period, only stop-loss or correlation break
                stop_loss = (
                    (position == 1 and z < -entry_threshold * 1.5) or
                    (position == -1 and z > entry_threshold * 1.5) or
                    days_held > max_hold_days
                )
                if stop_loss:
                    position = 0
                    entry_date_idx = None
                    entry_z = None
        
        signals.iloc[i] = position
    
    return signals


class HybridSystem:
   
    def __init__(self):
        self.kalman = KalmanFilter()
        self.regime_classifier = RegimeClassifier()
        self.results = None
        
    def check_cointegration_health(self, stock_y, stock_x, window=60):
        recent_y = stock_y.iloc[-window:] if len(stock_y) >= window else stock_y
        recent_x = stock_x.iloc[-window:] if len(stock_x) >= window else stock_x
        
        try:
            _, p_value, _ = coint(recent_y, recent_x)
            
            if p_value > 0.10:
                return 'WARNING', p_value
            elif p_value > 0.05:
                return 'BROKEN', p_value
            else:
                return 'HEALTHY', p_value
        except:
            return 'ERROR', 1.0
        
    def run_backtest(self, stock_y, stock_x, market_index,
                     train_period=252,
                     entry_z_normal=2.0, exit_z_normal=0.5,
                     entry_z_volatile=2.5, exit_z_volatile=0.3,
                     min_hold_days=3,
                     min_correlation=0.5,
                     use_dynamic_sizing=True,
                     use_multi_timeframe=True,
                     use_coint_monitoring=True):

        # Align data
        data = pd.DataFrame({
            'Y': stock_y,
            'X': stock_x,
            'Market': market_index
        }).dropna()
        
        print(f"Backtest period: {data.index[0]} to {data.index[-1]}")
        print(f"Total trading days: {len(data)}")
        print(f"\n  SYSTEM FEATURES ENABLED:")
        print(f"   Adaptive Kalman filter (regime-dependent)")
        print(f"   3-tier regime classification")
        print(f"   MAD-based outlier detection")
        print(f"   Minimum holding period ({min_hold_days} days)")
        print(f"   Correlation stability filter (min {min_correlation})")
        if use_dynamic_sizing:
            print(f"   Dynamic position sizing")
        if use_multi_timeframe:
            print(f"   Multiple timeframe confirmation (30/60/90 days)")
        if use_coint_monitoring:
            print(f"   Cointegration health monitoring")
        print()
        
        # Step 1: Train regime classifier
        print("Training 3-tier regime classifier")
        prices_df = pd.DataFrame({'Y': stock_y, 'X': stock_x})
        features = self.regime_classifier.create_features(prices_df, market_index)
        labels = self.regime_classifier.label_regimes(features)
        
        train_features = features.iloc[:train_period]
        train_labels = labels.iloc[:train_period]
        
        self.regime_classifier.train(train_features, train_labels)
        
        # Step 2: Predict regimes
        data['regime'] = self.regime_classifier.predict(features)
        
        # Step 3: Run Kalman filter
        print("Running adaptive Kalman filter with outlier detection")
        spreads = []
        spread_stds = []
        hedge_ratios = []
        
        regime_map = {0: 'crisis', 1: 'volatile', 2: 'normal'}
        
        for i in range(len(data)):
            y = data['Y'].iloc[i]
            x = data['X'].iloc[i]
            regime_label = data['regime'].iloc[i]
            regime_str = regime_map.get(regime_label, 'normal')
            
            et, sqrt_Qt = self.kalman.update(y, x, regime=regime_str)
            spreads.append(et)
            spread_stds.append(sqrt_Qt)
            hedge_ratios.append(self.kalman.get_hedge_ratio())
        
        data['spread'] = spreads
        data['spread_std'] = spread_stds
        data['hedge_ratio'] = hedge_ratios
        
        # Step 4: Calculate z-scores
        print("Calculating robust z-scores...")
        data['z_score'] = calculate_robust_zscore(
            pd.Series(spreads, index=data.index),
            base_window=60
        )
        
        # Step 5: Cointegration health monitoring
        data['coint_health'] = 'HEALTHY'
        data['coint_pvalue'] = 0.0
        
        if use_coint_monitoring:
            print("Monitoring cointegration health")
            for i in range(60, len(data), 20):  # Check every 20 days
                health, p_value = self.check_cointegration_health(
                    data['Y'].iloc[:i],
                    data['X'].iloc[:i],
                    window=60
                )
                
                data.loc[data.index[i]:, 'coint_health'] = health
                data.loc[data.index[i]:, 'coint_pvalue'] = p_value
                
                if health == 'BROKEN':
                    print(f"   Cointegration broken at {data.index[i]} (p={p_value:.3f})")
        
        # Step 6: Generate ultimate signals
        print("Generating ultimate trading signals...")
        data['final_signal'] = generate_ultimate_signals(
            data['z_score'],
            data['regime'],
            data,
            entry_z_normal=entry_z_normal,
            exit_z_normal=exit_z_normal,
            entry_z_volatile=entry_z_volatile,
            exit_z_volatile=exit_z_volatile,
            min_hold_days=min_hold_days,
            min_correlation=min_correlation,
            use_multi_timeframe=use_multi_timeframe
        )
        
        # Force exit if cointegration broken
        if use_coint_monitoring:
            broken_mask = data['coint_health'] == 'BROKEN'
            data.loc[broken_mask, 'final_signal'] = 0
        
        # Step 7: Dynamic position sizing
        if use_dynamic_sizing:
            print("Calculating dynamic position sizes...")
            data['market_vol'] = data['Market'].pct_change().rolling(20).std() * np.sqrt(252)
            data['position_size'] = [
                calculate_position_size(z, vol)
                for z, vol in zip(data['z_score'], data['market_vol'])
            ]
        else:
            data['position_size'] = 1.0
        
        # Step 8: Calculate returns
        data['returns_Y'] = data['Y'].pct_change()
        data['returns_X'] = data['X'].pct_change()
        
        data['spread_return'] = (
            data['returns_Y'] - data['hedge_ratio'] * data['returns_X']
        )
        
        data['strategy_return'] = (
            data['final_signal'].shift(1) * 
            data['position_size'] * 
            data['spread_return']
        )
        
        # Baseline: simple strategy without all the improvements
        simple_signals = generate_ultimate_signals(
            data['z_score'],
            pd.Series(2, index=data.index),  # Always "normal"
            data,
            entry_z_normal=2.0,
            exit_z_normal=0.5,
            min_hold_days=0,
            min_correlation=0.0,
            use_multi_timeframe=False
        )
        
        data['baseline_return'] = simple_signals.shift(1) * data['spread_return']
        
        # Cumulative returns
        data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
        data['baseline_cumulative'] = (1 + data['baseline_return']).cumprod()
        
        self.results = data
        
        print("\n Backtest complete!")
        return data
    
    def get_performance_metrics(self):
        if self.results is None:
            raise ValueError("Must run backtest first")
        
        data = self.results
        
        strategy_returns = data['strategy_return'].dropna()
        baseline_returns = data['baseline_return'].dropna()
        
        metrics = {
            'Ultimate Strategy': self._calculate_metrics(strategy_returns),
            'Baseline (Simple)': self._calculate_metrics(baseline_returns)
        }
        
        total_days = len(data)
        normal_days = (data['regime'] == 2).sum()
        volatile_days = (data['regime'] == 1).sum()
        crisis_days = (data['regime'] == 0).sum()
        
        # Cointegration health stats
        healthy_days = (data['coint_health'] == 'HEALTHY').sum()
        warning_days = (data['coint_health'] == 'WARNING').sum()
        broken_days = (data['coint_health'] == 'BROKEN').sum()
        
        metrics['System Statistics'] = {
            'Total Trading Days': total_days,
            'Normal Regime Days': normal_days,
            'Volatile Regime Days': volatile_days,
            'Crisis Regime Days': crisis_days,
            'Coint Healthy Days': healthy_days,
            'Coint Warning Days': warning_days,
            'Coint Broken Days': broken_days,
            'Crisis Filter Rate': f"{(crisis_days/total_days)*100:.1f}%",
            'Coint Protection Rate': f"{(broken_days/total_days)*100:.1f}%"
        }
        
        return metrics
    
    def get_trade_analysis(self):
        if self.results is None:
            raise ValueError("Must run backtest first")
        
        data = self.results.copy()
        data['position_change'] = data['final_signal'].diff()
        
        trades = []
        current_trade = None
        
        for i in range(len(data)):
            if data['position_change'].iloc[i] != 0 and data['final_signal'].iloc[i] != 0:
                if current_trade is None:
                    current_trade = {
                        'entry_date': data.index[i],
                        'entry_signal': data['final_signal'].iloc[i],
                        'entry_z': data['z_score'].iloc[i],
                        'entry_hedge': data['hedge_ratio'].iloc[i],
                        'entry_regime': data['regime'].iloc[i],
                        'entry_pos_size': data['position_size'].iloc[i],
                        'entry_coint_health': data['coint_health'].iloc[i]
                    }
            
            elif data['position_change'].iloc[i] != 0 and data['final_signal'].iloc[i] == 0:
                if current_trade is not None:
                    exit_idx = i
                    entry_idx = data.index.get_loc(current_trade['entry_date'])
                    
                    trade_returns = data['strategy_return'].iloc[entry_idx:exit_idx+1]
                    trade_pnl = trade_returns.sum()
                    
                    current_trade.update({
                        'exit_date': data.index[i],
                        'exit_z': data['z_score'].iloc[i],
                        'exit_regime': data['regime'].iloc[i],
                        'exit_coint_health': data['coint_health'].iloc[i],
                        'duration_days': (data.index[i] - current_trade['entry_date']).days,
                        'pnl': trade_pnl,
                        'pnl_pct': trade_pnl * 100,
                        'profitable': trade_pnl > 0
                    })
                    
                    trades.append(current_trade)
                    current_trade = None
        
        return pd.DataFrame(trades)
    
    def _calculate_metrics(self, returns):
        if len(returns) == 0:
            return {
                'Total Return': '0.00%',
                'Annual Return': '0.00%',
                'Annual Volatility': '0.00%',
                'Sharpe Ratio': '0.00',
                'Max Drawdown': '0.00%',
                'Win Rate': '0.0%',
                'Total Trades': 0
            }
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            'Total Return': f"{total_return*100:.2f}%",
            'Annual Return': f"{annual_return*100:.2f}%",
            'Annual Volatility': f"{annual_vol*100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown*100:.2f}%",
            'Win Rate': f"{win_rate*100:.1f}%",
            'Total Trades': len(returns[returns != 0])
        }

# PART 6: DATA DOWNLOAD

def download_data(ticker_y, ticker_x, market_ticker='SPY',
                 start_date='2018-01-01', end_date='2024-01-01'):
    print(f"Downloading data for {ticker_y}, {ticker_x}, and {market_ticker}...")
    
    data_y = yf.download(ticker_y, start=start_date, end=end_date, progress=False, auto_adjust=True)
    data_x = yf.download(ticker_x, start=start_date, end=end_date, progress=False, auto_adjust=True)
    data_market = yf.download(market_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    def extract_close(df):
        if df.empty:
            raise ValueError("Downloaded data is empty")
        
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                return df['Close'].iloc[:, 0].squeeze()
            else:
                return df.iloc[:, 0].squeeze()
        else:
            if 'Close' in df.columns:
                return df['Close'].squeeze()
            else:
                return df.iloc[:, 0].squeeze()
    
    stock_y = extract_close(data_y)
    stock_x = extract_close(data_x)
    market = extract_close(data_market)
    
    if not isinstance(stock_y, pd.Series):
        stock_y = pd.Series(stock_y)
    if not isinstance(stock_x, pd.Series):
        stock_x = pd.Series(stock_x)
    if not isinstance(market, pd.Series):
        market = pd.Series(market)
    
    print(f" Downloaded {len(stock_y)} days of data")
    print(f"Date range: {stock_y.index[0]} to {stock_y.index[-1]}\n")
    
    return stock_y, stock_x, market

if __name__ == "__main__":
    print("\n")
    print("\n")
    print(" HYBRID PAIRS TRADING SYSTEM")
    print("="*20)
    print("""
    ALL FEATURES IMPLEMENTED:
    1.  Adaptive Kalman filter (regime-dependent process noise)
    2.  3-tier regime classification (NORMAL/VOLATILE/CRISIS)
    3.  Robust outlier handling (MAD-based detection)
    4.  Minimum holding period (3 days default)
    5.  Adaptive z-score window (60 days with expanding fallback)
    6.  Cointegration health monitoring (exits if breaks)
    7.  Correlation stability filter (min 0.5 default)
    8.  Dynamic position sizing (based on signal + volatility)
    9.  Multiple timeframe confirmation (30/60/90 days)
    10.  Half-life based exit timing (dynamic max hold)
    """)
    
    # Test on GOOD pair (critical!)
    print("Testing on BAC vs PNC (Quality Score: 110)...")
    print("-" * 80)
    
    bac, pnc, spy = download_data('BAC', 'PNC', 'SPY', '2018-01-01', '2024-01-01')
    
    system = HybridSystem()
    results = system.run_backtest(
        bac, pnc, spy,
        train_period=252,
        entry_z_normal=2.0,
        exit_z_normal=0.5,
        entry_z_volatile=2.5,
        exit_z_volatile=0.3,
        min_hold_days=3,
        min_correlation=0.5,
        use_dynamic_sizing=True,
        use_multi_timeframe=True,
        use_coint_monitoring=True
    )
    
    print("\n")
    print("\n")
    print("\n")
    print(" PERFORMANCE METRICS")
    print("="*20)
    
    metrics = system.get_performance_metrics()
    
    for strategy_name, strategy_metrics in metrics.items():
        if strategy_name != 'System Statistics':
            print(f"\n{strategy_name}:")
            print("-" * 40)
            for metric, value in strategy_metrics.items():
                print(f"  {metric:.<30} {value}")
    
    print(f"\n{'System Statistics':}")
    print("-" * 30)
    for key, value in metrics['System Statistics'].items():
        print(f"  {key:.<30} {value}")
    
    # Trade analysis
    trades = system.get_trade_analysis()
    if len(trades) > 0:
        print(f"\n")
        print(f"\n")
        print(f"\n")
        print(" TRADE ANALYSIS")
        print("="*20)
        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {trades['profitable'].mean()*100:.1f}%")
        print(f"Avg Trade P&L: {trades['pnl_pct'].mean():.2f}%")
        print(f"Avg Duration: {trades['duration_days'].mean():.1f} days")
        print(f"Avg Position Size: {trades['entry_pos_size'].mean():.2f}x")
        
        print(f"\nTrades by Entry Regime:")
        regime_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}
        for regime_code, regime_name in regime_names.items():
            regime_trades = trades[trades['entry_regime'] == regime_code]
            if len(regime_trades) > 0:
                print(f"  {regime_name}: {len(regime_trades)} trades, "
                      f"Win Rate: {regime_trades['profitable'].mean()*100:.1f}%, "
                      f"Avg P&L: {regime_trades['pnl_pct'].mean():.2f}%")
        
        print(f"\nTrades by Cointegration Health:")
        for health in ['HEALTHY', 'WARNING', 'BROKEN']:
            health_trades = trades[trades['entry_coint_health'] == health]
            if len(health_trades) > 0:
                print(f"  {health}: {len(health_trades)} trades, "
                      f"Win Rate: {health_trades['profitable'].mean()*100:.1f}%")
    
    # Improvement analysis
    print(f"\n")
    print(f"\n")
    print(f"\n")
    print(" IMPROVEMENTS OVER BASELINE")
    print("="*20)
    
    ultimate_sharpe = float(metrics['Ultimate Strategy']['Sharpe Ratio'])
    baseline_sharpe = float(metrics['Baseline (Simple)']['Sharpe Ratio'])
    
    if baseline_sharpe != 0:
        improvement = ((ultimate_sharpe - baseline_sharpe) / abs(baseline_sharpe)) * 100
        print(f"Sharpe Ratio Improvement: {improvement:+.1f}%")
    
    ultimate_return = float(metrics['Ultimate Strategy']['Total Return'].rstrip('%'))
    baseline_return = float(metrics['Baseline (Simple)']['Total Return'].rstrip('%'))
    print(f"Total Return: {ultimate_return:.1f}% vs {baseline_return:.1f}%")