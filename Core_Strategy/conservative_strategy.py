import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class ConservativeKalman:
    
    def __init__(self, delta=1e-4, var_e=1e-3, var_eta=1e-4):
        self.delta = delta
        self.var_e = var_e
        self.var_eta = var_eta
        self.wt = None
        self.Ct = None
        self.At = None
        self.Rt = None
        self.innovation_history = []
        
    def initialize(self, y0, x0):
        self.wt = y0 / x0 if x0 != 0 else 0
        self.Ct = self.delta
        
    def update(self, y, x, regime='normal'):
        if self.wt is None:
            self.initialize(y, x)
            return 0, np.sqrt(self.var_e)
        
        # Regime-dependent variance
        var_eta_adjusted = {
            'crisis': self.var_eta * 5,
            'volatile': self.var_eta * 2,
            'normal': self.var_eta
        }.get(regime, self.var_eta)
        
        # Prediction
        wt_minus = self.wt
        self.At = self.Ct + var_eta_adjusted
        
        # Innovation
        et = y - wt_minus * x
        self.Rt = self.At * x**2 + self.var_e
        
        # Track innovation history
        self.innovation_history.append(et)
        if len(self.innovation_history) > 50:
            self.innovation_history.pop(0)
        
        # Robust outlier detection using MAD (Median Absolute Deviation)
        if len(self.innovation_history) >= 10:
            innovations = np.array(self.innovation_history)
            median = np.median(innovations)
            mad = np.median(np.abs(innovations - median))
            
            # Outlier if > 3 MAD from median
            if abs(et - median) > 3 * mad:
                Kt = 0.05 * (self.At * x / self.Rt if self.Rt != 0 else 0)
            else:
                Kt = self.At * x / self.Rt if self.Rt != 0 else 0
        else:
            Kt = self.At * x / self.Rt if self.Rt != 0 else 0
        
        # Update
        self.wt = wt_minus + Kt * et
        self.Ct = self.At - Kt * x * self.At
        
        sqrt_Qt = np.sqrt(self.Rt)
        
        return et, sqrt_Qt
    
    def get_hedge_ratio(self):
        return self.wt if self.wt is not None else 0


class StrictRegimeClassifier:
    
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
        
        # Volatility trend
        features['vol_trend'] = features['vol_5'] - features['vol_60']
        features['vol_accel'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        
        # Drawdown
        for window in [10, 20, 40, 60]:
            rolling_max = market_index.rolling(window, min_periods=1).max()
            features[f'dd_{window}'] = (market_index - rolling_max) / rolling_max
        
        # Returns
        for window in [5, 10, 20]:
            features[f'ret_{window}'] = market_index.pct_change(window)
        
        # Trend
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
            
            # Pair spread volatility
            pair_ratio = prices.iloc[:, 0] / prices.iloc[:, 1]
            features['pair_vol'] = pair_ratio.pct_change().rolling(20).std() * np.sqrt(252)
            features['pair_vol_spike'] = features['pair_vol'] / features['pair_vol'].rolling(60).mean()
        
        return features.ffill().fillna(0)
    
    def label_regimes(self, features, 
                     vol_crisis=0.25, vol_volatile=0.18,
                     dd_crisis=-0.12, dd_volatile=-0.06):
        labels = pd.Series(2, index=features.index)  # Default: NORMAL
        
        # Mark VOLATILE 
        volatile_conditions = (
            (features['vol_20'] > vol_volatile) |
            (features['dd_60'] < dd_volatile) |
            (features.get('vol_accel', 1) > 1.5) |
            (features.get('pair_vol_spike', 1) > 1.3)
        )
        labels[volatile_conditions] = 1
        
        # Mark CRISIS (overrides volatile)
        crisis_conditions = (
            (features['vol_20'] > vol_crisis) |
            (features['dd_60'] < dd_crisis) |
            (features.get('vol_accel', 1) > 2.0) |
            (features.get('pair_corr', 1) < 0.2)
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
        
        print(f"Regime classifier trained on {len(X)} samples")
        print(f"NORMAL: {(y==2).sum()}, VOLATILE: {(y==1).sum()}, CRISIS: {(y==0).sum()}")
        
    def predict(self, features):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(features.fillna(0))
        predictions = self.model.predict(X_scaled)
        return pd.Series(predictions, index=features.index)


def calculate_ultra_conservative_zscore(spread, window=60):
    # Use expanding window for first 60 days
    if len(spread) < window:
        mean = spread.expanding(min_periods=10).mean()
        std = spread.expanding(min_periods=10).std()
    else:
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
    
    z_score = (spread - mean) / (std + 1e-8)
    return z_score


def generate_ultra_conservative_signals(z_scores, regimes, spread_returns,
                                        entry_z_normal=2.5, exit_z_normal=0.3,
                                        entry_z_volatile=3.0, exit_z_volatile=0.2,
                                        min_hold_days=3):

    signals = pd.Series(0, index=z_scores.index)
    position = 0
    entry_date_idx = None
    entry_z = None
    
    for i in range(len(z_scores)):
        z = z_scores.iloc[i]
        regime = regimes.iloc[i]
        
        if pd.isna(z) or pd.isna(regime):
            signals.iloc[i] = position
            continue
        
        # NEVER trade in CRISIS
        if regime == 0:
            position = 0
            entry_date_idx = None
            entry_z = None
            signals.iloc[i] = position
            continue
        
        # Adjust thresholds
        if regime == 1:  # VOLATILE
            entry_threshold = entry_z_volatile
            exit_threshold = exit_z_volatile
        else:  # NORMAL
            entry_threshold = entry_z_normal
            exit_threshold = exit_z_normal
        
        # Entry logic
        if position == 0:
            # Only enter on EXTREME z-scores
            if z > entry_threshold:
                # Additional filter: check if spread is still moving in our direction
                if i >= 5:
                    recent_z_trend = z_scores.iloc[i-5:i].mean()
                    if z > recent_z_trend:  # Z-score is increasing (good for short)
                        position = -1
                        entry_date_idx = i
                        entry_z = z
                else:
                    position = -1
                    entry_date_idx = i
                    entry_z = z
                    
            elif z < -entry_threshold:
                if i >= 5:
                    recent_z_trend = z_scores.iloc[i-5:i].mean()
                    if z < recent_z_trend:  # Z-score is decreasing (good for long)
                        position = 1
                        entry_date_idx = i
                        entry_z = z
                else:
                    position = 1
                    entry_date_idx = i
                    entry_z = z
        
        # Exit logic
        elif position != 0:
            days_held = i - entry_date_idx if entry_date_idx is not None else 0
            
            # Force exit conditions
            exit_conditions = [
                # Normal mean reversion
                abs(z) < exit_threshold,
                
                # Stop-loss: z-score moved against us too much
                (position == 1 and z < -entry_threshold * 1.3),
                (position == -1 and z > entry_threshold * 1.3),
                
                # Regime change to volatile (if we entered in normal)
                (regime == 1 and entry_z is not None and abs(entry_z) < entry_z_volatile),
                
                # Maximum holding period (mean reversion failed)
                days_held > 20
            ]
            
            # Apply minimum holding period before allowing normal exit
            if days_held >= min_hold_days:
                if any(exit_conditions):
                    position = 0
                    entry_date_idx = None
                    entry_z = None
            else:
                # Before min holding, only stop-loss exits allowed
                if (position == 1 and z < -entry_threshold * 1.3) or \
                   (position == -1 and z > entry_threshold * 1.3) or \
                   days_held > 20:
                    position = 0
                    entry_date_idx = None
                    entry_z = None
        
        signals.iloc[i] = position
    
    return signals


class ConservativeSystem:
    
    def __init__(self):
        self.kalman = ConservativeKalman()
        self.regime_classifier = StrictRegimeClassifier()
        self.results = None
        
    def run_backtest(self, stock_y, stock_x, market_index, 
                     train_period=252, 
                     entry_z_normal=2.5, exit_z_normal=0.3,
                     entry_z_volatile=3.0, exit_z_volatile=0.2,
                     min_hold_days=3,
                     z_score_window=60):
        data = pd.DataFrame({
            'Y': stock_y,
            'X': stock_x,
            'Market': market_index
        }).dropna()
        
        print(f"Backtest period: {data.index[0]} to {data.index[-1]}")
        print(f"Total trading days: {len(data)}")
        
        # Preliminary regime detection
        prices_df = pd.DataFrame({'Y': stock_y, 'X': stock_x})
        features = self.regime_classifier.create_features(prices_df, market_index)
        labels = self.regime_classifier.label_regimes(features)
        
        # Run Kalman filter
        spreads = []
        spread_stds = []
        hedge_ratios = []
        
        for i in range(len(data)):
            y = data['Y'].iloc[i]
            x = data['X'].iloc[i]
            regime_label = labels.iloc[i] if i < len(labels) else 2
            
            regime_str = 'crisis' if regime_label == 0 else ('volatile' if regime_label == 1 else 'normal')
            
            et, sqrt_Qt = self.kalman.update(y, x, regime=regime_str)
            spreads.append(et)
            spread_stds.append(sqrt_Qt)
            hedge_ratios.append(self.kalman.get_hedge_ratio())
        
        data['spread'] = spreads
        data['spread_std'] = spread_stds
        data['hedge_ratio'] = hedge_ratios
        
        # Train regime classifier
        train_features = features.iloc[:train_period]
        train_labels = labels.iloc[:train_period]
        
        self.regime_classifier.train(train_features, train_labels)
        data['regime'] = self.regime_classifier.predict(features)
        
        # Calculate z-scores with longer window
        data['z_score'] = calculate_ultra_conservative_zscore(
            pd.Series(spreads, index=data.index),
            window=z_score_window
        )
        
        # Generate signals
        data['spread_return'] = pd.Series(spreads, index=data.index).pct_change()
        
        data['final_signal'] = generate_ultra_conservative_signals(
            data['z_score'], 
            data['regime'],
            data['spread_return'],
            entry_z_normal=entry_z_normal,
            exit_z_normal=exit_z_normal,
            entry_z_volatile=entry_z_volatile,
            exit_z_volatile=exit_z_volatile,
            min_hold_days=min_hold_days
        )
        
        # Calculate returns
        data['returns_Y'] = data['Y'].pct_change()
        data['returns_X'] = data['X'].pct_change()
        
        data['spread_return'] = (
            data['returns_Y'] - data['hedge_ratio'] * data['returns_X']
        )
        
        data['strategy_return'] = data['final_signal'].shift(1) * data['spread_return']
        
        # Math-only baseline (same thresholds but no regime filter)
        baseline_signals = generate_ultra_conservative_signals(
            data['z_score'],
            pd.Series(2, index=data.index),  # Always "normal"
            data['spread_return'],
            entry_z_normal=entry_z_normal,
            exit_z_normal=exit_z_normal,
            min_hold_days=min_hold_days
        )
        data['math_only_return'] = baseline_signals.shift(1) * data['spread_return']
        
        # Cumulative
        data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
        data['math_only_cumulative'] = (1 + data['math_only_return']).cumprod()
        
        self.results = data
        return data
    
    def get_performance_metrics(self):
        if self.results is None:
            raise ValueError("Must run backtest first")
        
        data = self.results
        
        strategy_returns = data['strategy_return'].dropna()
        math_only_returns = data['math_only_return'].dropna()
        
        metrics = {
            'Hybrid Strategy': self._calculate_metrics(strategy_returns),
            'Math Only': self._calculate_metrics(math_only_returns)
        }
        
        total_days = len(data)
        normal_days = (data['regime'] == 2).sum()
        volatile_days = (data['regime'] == 1).sum()
        crisis_days = (data['regime'] == 0).sum()
        
        metrics['Additional Info'] = {
            'Total Trading Days': total_days,
            'Normal Regime Days': normal_days,
            'Volatile Regime Days': volatile_days,
            'Crisis Regime Days': crisis_days,
            'Crisis Filter Rate': f"{(crisis_days/total_days)*100:.1f}%"
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
                        'entry_regime': data['regime'].iloc[i]
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
                        'duration_days': (data.index[i] - current_trade['entry_date']).days,
                        'pnl': trade_pnl,
                        'pnl_pct': trade_pnl * 100,
                        'profitable': trade_pnl > 0
                    })
                    
                    trades.append(current_trade)
                    current_trade = None
        
        return pd.DataFrame(trades)
    
    def _calculate_metrics(self, returns):
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
    
    print(f"Downloaded {len(stock_y)} days of data")
    
    return stock_y, stock_x, market


if __name__ == "__main__":
    print("\n")
    print("\n")
    print("CONSERVATIVE PAIRS TRADING SYSTEM")
    print("\n")
    print("\n")
    
    pep, ko, spy = download_data('PEP', 'KO', 'SPY', '2018-01-01', '2024-01-01')
    
    system = ConservativeSystem()
    results = system.run_backtest(
        pep, ko, spy,
        train_period=252,
        entry_z_normal=2.5,      # Higher threshold
        exit_z_normal=0.3,       # Tighter exit
        entry_z_volatile=3.0,    # Even higher in volatile
        exit_z_volatile=0.2,     # Very tight exit
        min_hold_days=3,         # Enforce minimum duration
        z_score_window=60        # Longer window
    )
    
    metrics = system.get_performance_metrics()
    
    print("\n")
    print("\n")
    print("PERFORMANCE METRICS")
    print("\n")
    print("\n")
    
    for strategy_name, strategy_metrics in metrics.items():
        if strategy_name != 'Additional Info':
            print(f"\n{strategy_name}:")
            print("-" * 40)
            for metric, value in strategy_metrics.items():
                print(f"  {metric:.<30} {value}")
    
    print(f"\n{'Additional Information':}")
    print("-" * 40)
    for key, value in metrics['Additional Info'].items():
        print(f"  {key:.<30} {value}")
    
    trades = system.get_trade_analysis()
    if len(trades) > 0:
        print(f"\n Trade Analysis:")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Win Rate: {trades['profitable'].mean()*100:.1f}%")
        print(f"  Avg Trade P&L: {trades['pnl_pct'].mean():.2f}%")
        print(f"  Avg Duration: {trades['duration_days'].mean():.1f} days")