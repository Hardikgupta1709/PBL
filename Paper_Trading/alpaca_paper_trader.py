"""
Paper Trading System - Integrated with Alpaca
Automated daily trading for validated SO/SRE pair
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api import REST
import config
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Core_Strategy.ultra_conservative_strategy import UltraConservativeSystem, download_data


class AlpacaPaperTrader:
    """
    Automated paper trading system for validated pairs
    """
    
    def __init__(self):
        # Initialize Alpaca API
        self.api = REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL
        )
        
        self.pairs = config.VALIDATED_PAIRS
        self.log_file = config.TRADE_LOG_FILE
        
    def log_message(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(config.LOG_FILE, 'a') as f:
            f.write(log_msg + '\n')
    
    def get_account_summary(self):
        """Get current account status"""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }
    
    def get_current_positions(self):
        """Get current positions"""
        positions = self.api.list_positions()
        return {pos.symbol: float(pos.qty) for pos in positions}
    
    def get_current_price(self, symbol):
        """Get latest price for symbol"""
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except:
            # Fallback to last quote
            quote = self.api.get_latest_quote(symbol)
            return float(quote.ask_price)
    
    def calculate_signals(self, pair_name, pair_config):
        """Calculate trading signals for a pair"""
        
        ticker_y = pair_config['ticker_y']
        ticker_x = pair_config['ticker_x']
        
        self.log_message(f"Calculating signals for {pair_name} ({ticker_y}/{ticker_x})...")
        
        try:
            # Download recent data (1 year + buffer)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)
            
            stock_y, stock_x, market = download_data(
                ticker_y, ticker_x, 'SPY',
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Run strategy
            system = UltraConservativeSystem()
            results = system.run_backtest(
                stock_y, stock_x, market,
                train_period=min(252, len(stock_y) // 2),
                entry_z_normal=pair_config['entry_z_normal'],
                exit_z_normal=pair_config['exit_z_normal'],
                entry_z_volatile=pair_config['entry_z_volatile'],
                exit_z_volatile=pair_config['exit_z_volatile'],
                min_hold_days=pair_config['min_hold_days'],
                z_score_window=pair_config['z_score_window']
            )
            
            # Get latest signal
            latest_signal = int(results['final_signal'].iloc[-1])
            latest_z = float(results['z_score'].iloc[-1])
            latest_regime = int(results['regime'].iloc[-1])
            hedge_ratio = float(results['hedge_ratio'].iloc[-1])
            
            regime_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}
            
            return {
                'signal': latest_signal,
                'z_score': latest_z,
                'regime': regime_names.get(latest_regime, 'UNKNOWN'),
                'hedge_ratio': hedge_ratio,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.log_message(f"❌ Error calculating signals: {e}")
            return None
    
    def execute_order(self, symbol, qty, side):
        """Execute order via Alpaca"""
        try:
            if qty == 0:
                return None
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(int(qty)),
                side=side,
                type='market',
                time_in_force='day'
            )
            
            self.log_message(f"  ✅ Order: {side.upper()} {abs(int(qty))} {symbol}")
            return order
            
        except Exception as e:
            self.log_message(f"  ❌ Order failed for {symbol}: {e}")
            return None
    
    def manage_pair(self, pair_name, pair_config, signal_data):
        """Manage position for a pair"""
        
        ticker_y = pair_config['ticker_y']
        ticker_x = pair_config['ticker_x']
        
        # Get current state
        current_positions = self.get_current_positions()
        current_y = current_positions.get(ticker_y, 0)
        current_x = current_positions.get(ticker_x, 0)
        
        signal = signal_data['signal']
        hedge_ratio = signal_data['hedge_ratio']
        
        self.log_message(f"\n📊 {pair_name}:")
        self.log_message(f"  Signal: {signal} | Z-score: {signal_data['z_score']:.2f} | Regime: {signal_data['regime']}")
        self.log_message(f"  Current: Y={current_y}, X={current_x}")
        
        # Calculate target positions
        account = self.get_account_summary()
        position_value = account['portfolio_value'] * pair_config['position_size']
        
        # Get current prices
        price_y = self.get_current_price(ticker_y)
        price_x = self.get_current_price(ticker_x)
        
        self.log_message(f"  Prices: {ticker_y}=${price_y:.2f}, {ticker_x}=${price_x:.2f}")
        
        # Calculate targets based on signal
        if signal == 1:  # LONG spread
            target_y = int(position_value / price_y)
            target_x = -int(target_y * hedge_ratio)
            action = "LONG SPREAD (Buy Y, Sell X)"
            
        elif signal == -1:  # SHORT spread
            target_y = -int(position_value / price_y)
            target_x = int(abs(target_y) * hedge_ratio)
            action = "SHORT SPREAD (Sell Y, Buy X)"
            
        else:  # Flat
            target_y = 0
            target_x = 0
            action = "FLAT (Close all)"
        
        self.log_message(f"  Action: {action}")
        self.log_message(f"  Target: Y={target_y}, X={target_x}")
        
        # Execute trades
        delta_y = target_y - current_y
        delta_x = target_x - current_x
        
        if abs(delta_y) > 0:
            side_y = 'buy' if delta_y > 0 else 'sell'
            self.execute_order(ticker_y, abs(delta_y), side_y)
        
        if abs(delta_x) > 0:
            side_x = 'buy' if delta_x > 0 else 'sell'
            self.execute_order(ticker_x, abs(delta_x), side_x)
        
        # Log trade
        self.log_trade(pair_name, signal_data, target_y, target_x, price_y, price_x)
    
    def log_trade(self, pair_name, signal_data, qty_y, qty_x, price_y, price_x):
        """Log trade to CSV"""
        
        log_entry = {
            'timestamp': datetime.now(),
            'pair': pair_name,
            'signal': signal_data['signal'],
            'z_score': signal_data['z_score'],
            'regime': signal_data['regime'],
            'qty_y': qty_y,
            'qty_x': qty_x,
            'price_y': price_y,
            'price_x': price_x,
            'hedge_ratio': signal_data['hedge_ratio']
        }
        
        df = pd.DataFrame([log_entry])
        
        if os.path.exists(self.log_file):
            df.to_csv(self.log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.log_file, index=False)
    
    def run_daily_update(self):
        """Main daily update function"""
        
        self.log_message("="*80)
        self.log_message("DAILY PAPER TRADING UPDATE")
        self.log_message("="*80)
        
        # Check if market is open
        clock = self.api.get_clock()
        if clock.is_open:
            self.log_message("⚠️ Market is still open. Best to run after 4 PM ET.")
        
        # Account summary
        account = self.get_account_summary()
        self.log_message(f"\n💰 Account Summary:")
        self.log_message(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        self.log_message(f"  Cash: ${account['cash']:,.2f}")
        self.log_message(f"  Buying Power: ${account['buying_power']:,.2f}")
        
        # Process each pair
        for pair_name, pair_config in self.pairs.items():
            try:
                # Calculate signals
                signal_data = self.calculate_signals(pair_name, pair_config)
                
                if signal_data is None:
                    self.log_message(f"\n⚠️ Skipping {pair_name} - signal calculation failed")
                    continue
                
                # Manage position
                self.manage_pair(pair_name, pair_config, signal_data)
                
            except Exception as e:
                self.log_message(f"\n❌ Error processing {pair_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.log_message("\n" + "="*80)
        self.log_message("✅ Update complete!")
        self.log_message("="*80 + "\n")
    
    def get_performance_summary(self):
        """Generate performance summary"""
        
        if not os.path.exists(self.log_file):
            self.log_message("No trades logged yet")
            return
        
        trades = pd.read_csv(self.log_file)
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        
        self.log_message("\n" + "="*80)
        self.log_message("PERFORMANCE SUMMARY")
        self.log_message("="*80)
        
        self.log_message(f"\n📅 Period: {trades['timestamp'].min()} to {trades['timestamp'].max()}")
        self.log_message(f"   Days: {(trades['timestamp'].max() - trades['timestamp'].min()).days}")
        self.log_message(f"   Updates: {len(trades)}")
        
        # By pair
        for pair in trades['pair'].unique():
            pair_trades = trades[trades['pair'] == pair]
            
            self.log_message(f"\n📊 {pair}:")
            self.log_message(f"   Total signals: {len(pair_trades)}")
            self.log_message(f"   Long signals: {(pair_trades['signal'] == 1).sum()}")
            self.log_message(f"   Short signals: {(pair_trades['signal'] == -1).sum()}")
            self.log_message(f"   Flat signals: {(pair_trades['signal'] == 0).sum()}")
            self.log_message(f"   Avg |Z-score|: {pair_trades['z_score'].abs().mean():.2f}")
        
        # Current account
        account = self.get_account_summary()
        initial_value = 100000
        pnl = account['portfolio_value'] - initial_value
        return_pct = (pnl / initial_value) * 100
        
        self.log_message(f"\n💰 Account Performance:")
        self.log_message(f"   Initial Value: ${initial_value:,.2f}")
        self.log_message(f"   Current Value: ${account['portfolio_value']:,.2f}")
        self.log_message(f"   P&L: ${pnl:,.2f} ({return_pct:+.2f}%)")


if __name__ == "__main__":
    trader = AlpacaPaperTrader()
    
    # Run daily update
    trader.run_daily_update()
    
    # Show performance
    trader.get_performance_summary()