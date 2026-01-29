"""
Optimized Alpaca Paper Trading System
Integrated with Ultra-Conservative Strategy
Uses modern alpaca-py SDK for best performance
"""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
import os
import sys
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


import Paper_Trading.config as config

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=log_format,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlpacaPaperTrader:
    """
    High-performance paper trading system with:
    - Modern alpaca-py SDK integration
    - Strategy system integration
    - Comprehensive error handling
    - Position management
    - Performance tracking
    """
    
    def __init__(self):
        """Initialize the trading system"""
        
        logger.info("Initializing Alpaca Paper Trader")
        
        # Initialize API clients
        self.trading_client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.PAPER_TRADING
        )
        
        self.data_client = StockHistoricalDataClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY
        )
        
        # Configuration
        self.pairs = config.VALIDATED_PAIRS
        self.log_file = config.TRADE_LOG_FILE
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Cache for prices
        self._price_cache = {}
        self._cache_timeout = 60  # seconds
        
        logger.info(f"Initialized with {len(self.pairs)} pair(s)")
    
    def log_message(self, message: str, level: str = 'info'):
        """
        Log message with timestamp
        
        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error', 'debug')
        """
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
    
    def get_account_summary(self) -> Dict:
        """
        Get current account status with caching
        
        Returns:
            Dictionary with account information
        """
        try:
            account = self.trading_client.get_account()
            
            summary = {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'initial_capital': config.INITIAL_CAPITAL,
                'total_return': (float(account.portfolio_value) - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL,
                'status': account.status,
                'trading_blocked': account.trading_blocked
            }
            
            return summary
            
        except APIError as e:
            self.log_message(f"Error fetching account: {e}", 'error')
            raise
    
    def get_current_positions(self) -> Dict[str, float]:
        """
        Get current positions efficiently
        
        Returns:
            Dictionary mapping symbols to quantities
        """
        try:
            positions = self.trading_client.get_all_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
            
        except APIError as e:
            self.log_message(f"Error fetching positions: {e}", 'error')
            return {}
    
    def get_current_price(self, symbol: str, use_cache: bool = True) -> float:
        """
        Get latest price for symbol with caching
        
        Args:
            symbol: Stock ticker
            use_cache: Whether to use cached price
            
        Returns:
            Current price
        """
        # Check cache
        if use_cache and symbol in self._price_cache:
            price, timestamp = self._price_cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < self._cache_timeout:
                return price
        
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trades = self.data_client.get_stock_latest_trade(request)
            price = float(trades[symbol].price)
            
            # Update cache
            self._price_cache[symbol] = (price, datetime.now())
            
            return price
            
        except Exception as e:
            self.log_message(f"Error fetching price for {symbol}: {e}", 'warning')
            raise
    
    def get_batch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get prices for multiple symbols efficiently
        
        Args:
            symbols: List of stock tickers
            
        Returns:
            Dictionary mapping symbols to prices
        """
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbols)
            trades = self.data_client.get_stock_latest_trade(request)
            
            prices = {}
            for symbol, trade in trades.items():
                price = float(trade.price)
                prices[symbol] = price
                self._price_cache[symbol] = (price, datetime.now())
            
            return prices
            
        except Exception as e:
            self.log_message(f"Error fetching batch prices: {e}", 'error')
            # Fallback to individual requests
            return {s: self.get_current_price(s, use_cache=False) for s in symbols}
    
    def calculate_signals(
        self,
        pair_name: str,
        pair_config: Dict
    ) -> Optional[Dict]:
        """
        Calculate trading signals using the strategy system
        
        Args:
            pair_name: Name of the pair
            pair_config: Configuration for the pair
            
        Returns:
            Dictionary with signal data or None if error
        """
        ticker_y = pair_config['ticker_y']
        ticker_x = pair_config['ticker_x']
        
        self.log_message(f"Calculating signals for {pair_name} ({ticker_y}/{ticker_x})")
        
        try:
            # Import strategy (lazy import to avoid circular dependencies)
            from Core_Strategy.ultra_conservative_strategy import (
                UltraConservativeSystem,
                download_data
            )
            
            # Download recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.DATA_LOOKBACK_DAYS)
            
            stock_y, stock_x, market = download_data(
                ticker_y,
                ticker_x,
                config.BENCHMARK_SYMBOL,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(stock_y) < config.MIN_DATA_POINTS:
                self.log_message(f"Insufficient data for {pair_name}: {len(stock_y)} points", 'warning')
                return None
            
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
            
            # Extract latest signal
            latest_signal = int(results['final_signal'].iloc[-1])
            latest_z = float(results['z_score'].iloc[-1])
            latest_regime = int(results['regime'].iloc[-1])
            hedge_ratio = float(results['hedge_ratio'].iloc[-1])
            
            regime_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}
            
            signal_data = {
                'signal': latest_signal,
                'z_score': latest_z,
                'regime': regime_names.get(latest_regime, 'UNKNOWN'),
                'hedge_ratio': hedge_ratio,
                'timestamp': datetime.now(),
                'confidence': abs(latest_z) / pair_config['entry_z_normal']  # Signal strength
            }
            
            self.log_message(
                f"Signal for {pair_name}: {latest_signal} "
                f"(Z={latest_z:.2f}, Regime={signal_data['regime']})"
            )
            
            return signal_data
            
        except Exception as e:
            self.log_message(f"Error calculating signals for {pair_name}: {e}", 'error')
            import traceback
            traceback.print_exc()
            return None
    
    def execute_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = None,
        limit_price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Execute order via Alpaca with error handling
        
        Args:
            symbol: Stock ticker
            qty: Quantity (absolute value)
            side: 'buy' or 'sell'
            order_type: Order type (defaults to config)
            limit_price: Limit price for limit orders
            
        Returns:
            Order object or None if failed
        """
        if qty == 0:
            return None
        
        order_type = order_type or config.ORDER_TYPE
        qty = abs(int(qty))
        
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce[config.TIME_IN_FORCE.upper()]
            
            if order_type == 'limit' and limit_price:
                order_data = LimitOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            else:
                order_data = MarketOrderRequest(
                    symbol=symbol.upper(),
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            
            order = self.trading_client.submit_order(order_data)
            
            self.log_message(
                f"✅ Order placed: {side.upper()} {qty} {symbol} "
                f"({order_type}) - ID: {order.id}"
            )
            
            return order
            
        except APIError as e:
            self.log_message(f"❌ Order failed for {symbol}: {e}", 'error')
            return None
    
    def manage_pair(
        self,
        pair_name: str,
        pair_config: Dict,
        signal_data: Dict
    ) -> bool:
        """
        Manage position for a pair based on signals
        
        Args:
            pair_name: Name of the pair
            pair_config: Pair configuration
            signal_data: Signal data from strategy
            
        Returns:
            True if successful, False otherwise
        """
        ticker_y = pair_config['ticker_y']
        ticker_x = pair_config['ticker_x']
        
        # Get current state
        current_positions = self.get_current_positions()
        current_y = current_positions.get(ticker_y, 0)
        current_x = current_positions.get(ticker_x, 0)
        
        signal = signal_data['signal']
        hedge_ratio = signal_data['hedge_ratio']
        
        self.log_message(f"\n{'='*60}")
        self.log_message(f"Managing {pair_name}")
        self.log_message(f"Signal: {signal} | Z-score: {signal_data['z_score']:.2f} | Regime: {signal_data['regime']}")
        self.log_message(f"Current positions: Y={current_y:.0f}, X={current_x:.0f}")
        
        # Get account info
        account = self.get_account_summary()
        
        # Check if we should trade
        if account['trading_blocked']:
            self.log_message("⚠️  Trading is blocked, skipping", 'warning')
            return False
        
        # Get current prices (batch for efficiency)
        try:
            prices = self.get_batch_prices([ticker_y, ticker_x])
            price_y = prices[ticker_y]
            price_x = prices[ticker_x]
        except Exception as e:
            self.log_message(f"Failed to get prices: {e}", 'error')
            return False
        
        self.log_message(f"Prices: {ticker_y}=${price_y:.2f}, {ticker_x}=${price_x:.2f}")
        
        # Calculate target positions
        position_value = account['portfolio_value'] * pair_config['position_size']
        
        if signal == 1:  # LONG spread
            target_y = int(position_value / price_y)
            target_x = -int(target_y * hedge_ratio)
            action = "LONG SPREAD (Buy Y, Short X)"
            
        elif signal == -1:  # SHORT spread
            target_y = -int(position_value / price_y)
            target_x = int(abs(target_y) * hedge_ratio)
            action = "SHORT SPREAD (Short Y, Buy X)"
            
        else:  # Flat
            target_y = 0
            target_x = 0
            action = "FLAT (Close all positions)"
        
        self.log_message(f"Action: {action}")
        self.log_message(f"Target positions: Y={target_y}, X={target_x}")
        
        # Calculate deltas
        delta_y = target_y - current_y
        delta_x = target_x - current_x
        
        self.log_message(f"Position changes: ΔY={delta_y:+.0f}, ΔX={delta_x:+.0f}")
        
        # Execute trades
        orders_placed = []
        
        if abs(delta_y) > 0:
            side_y = 'buy' if delta_y > 0 else 'sell'
            order_y = self.execute_order(ticker_y, abs(delta_y), side_y)
            if order_y:
                orders_placed.append(order_y)
        
        if abs(delta_x) > 0:
            side_x = 'buy' if delta_x > 0 else 'sell'
            order_x = self.execute_order(ticker_x, abs(delta_x), side_x)
            if order_x:
                orders_placed.append(order_x)
        
        # Log trade
        if orders_placed:
            self.log_trade(
                pair_name, signal_data,
                target_y, target_x,
                price_y, price_x,
                orders_placed
            )
        
        self.log_message(f"{'='*60}\n")
        
        return len(orders_placed) > 0
    
    def log_trade(
        self,
        pair_name: str,
        signal_data: Dict,
        qty_y: int,
        qty_x: int,
        price_y: float,
        price_x: float,
        orders: List
    ):
        """
        Log trade to CSV file
        
        Args:
            pair_name: Name of the pair
            signal_data: Signal data
            qty_y: Quantity of Y
            qty_x: Quantity of X
            price_y: Price of Y
            price_x: Price of X
            orders: List of order objects
        """
        try:
            log_entry = {
                'timestamp': datetime.now(),
                'pair': pair_name,
                'signal': signal_data['signal'],
                'z_score': signal_data['z_score'],
                'regime': signal_data['regime'],
                'confidence': signal_data.get('confidence', 0),
                'qty_y': qty_y,
                'qty_x': qty_x,
                'price_y': price_y,
                'price_x': price_x,
                'hedge_ratio': signal_data['hedge_ratio'],
                'orders_placed': len(orders),
                'order_ids': ','.join(o.id for o in orders)
            }
            
            df = pd.DataFrame([log_entry])
            
            if self.log_file.exists():
                df.to_csv(self.log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.log_file, index=False)
            
            self.log_message(f"Trade logged to {self.log_file.name}")
            
        except Exception as e:
            self.log_message(f"Error logging trade: {e}", 'error')
    
    def run_daily_update(self) -> bool:
        """
        Main daily update function
        
        Returns:
            True if update completed successfully
        """
        self.log_message("="*80)
        self.log_message("🚀 DAILY PAPER TRADING UPDATE")
        self.log_message("="*80)
        
        try:
            # Check market status
            clock = self.trading_client.get_clock()
            
            if clock.is_open:
                self.log_message("⚠️  Market is still open. Best to run after 4 PM ET.", 'warning')
            
            # Get account summary
            account = self.get_account_summary()
            
            self.log_message(f"\n💰 Account Summary:")
            self.log_message(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
            self.log_message(f"Cash: ${account['cash']:,.2f}")
            self.log_message(f"Buying Power: ${account['buying_power']:,.2f}")
            self.log_message(f"Total Return: {account['total_return']:+.2%}")
            
            # Check for issues
            if account['trading_blocked']:
                self.log_message("❌ Trading is blocked on this account!", 'error')
                return False
            
            # Process each pair
            success_count = 0
            
            for pair_name, pair_config in self.pairs.items():
                try:
                    self.log_message(f"\nProcessing {pair_name}...")
                    
                    # Calculate signals
                    signal_data = self.calculate_signals(pair_name, pair_config)
                    
                    if signal_data is None:
                        self.log_message(f"⚠️  Skipping {pair_name} - signal calculation failed", 'warning')
                        continue
                    
                    # Manage position
                    if self.manage_pair(pair_name, pair_config, signal_data):
                        success_count += 1
                    
                except Exception as e:
                    self.log_message(f"❌ Error processing {pair_name}: {e}", 'error')
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.log_message("\n" + "="*80)
            self.log_message(f"✅ Update complete! Processed {success_count}/{len(self.pairs)} pair(s)")
            self.log_message("="*80 + "\n")
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Critical error in daily update: {e}", 'error')
            import traceback
            traceback.print_exc()
            return False
    
    def get_performance_summary(self):
        """Generate and display performance summary"""
        
        if not self.log_file.exists():
            self.log_message("No trades logged yet")
            return
        
        try:
            trades = pd.read_csv(self.log_file)
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            
            self.log_message("\n" + "="*80)
            self.log_message("📊 PERFORMANCE SUMMARY")
            self.log_message("="*80)
            
            # Time period
            start_date = trades['timestamp'].min()
            end_date = trades['timestamp'].max()
            days_active = (end_date - start_date).days
            
            self.log_message(f"\n📅 Period: {start_date} to {end_date}")
            self.log_message(f"Days Active: {days_active}")
            self.log_message(f"Total Updates: {len(trades)}")
            
            # By pair analysis
            for pair in trades['pair'].unique():
                pair_trades = trades[trades['pair'] == pair]
                
                self.log_message(f"\n📊 {pair}:")
                self.log_message(f"  Total Signals: {len(pair_trades)}")
                self.log_message(f"  Long: {(pair_trades['signal'] == 1).sum()}")
                self.log_message(f"  Short: {(pair_trades['signal'] == -1).sum()}")
                self.log_message(f"  Flat: {(pair_trades['signal'] == 0).sum()}")
                self.log_message(f"  Avg |Z-score|: {pair_trades['z_score'].abs().mean():.2f}")
                self.log_message(f"  Avg Confidence: {pair_trades['confidence'].mean():.2f}")
            
            # Account performance
            account = self.get_account_summary()
            
            self.log_message(f"\n💰 Account Performance:")
            self.log_message(f"  Initial Value: ${config.INITIAL_CAPITAL:,.2f}")
            self.log_message(f"  Current Value: ${account['portfolio_value']:,.2f}")
            
            pnl = account['portfolio_value'] - config.INITIAL_CAPITAL
            pnl_pct = account['total_return'] * 100
            
            self.log_message(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            
            # Annualized return (if enough time has passed)
            if days_active > 30:
                annualized_return = (account['total_return'] * 365 / days_active) * 100
                self.log_message(f"  Annualized Return (est): {annualized_return:+.2f}%")
            
            self.log_message("\n" + "="*80)
            
        except Exception as e:
            self.log_message(f"Error generating performance summary: {e}", 'error')
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


def main():
    """Main execution function"""
    
    # Validate configuration
    if not config.validate_config():
        logger.error("Configuration validation failed!")
        sys.exit(1)
    
    # Create trader
    trader = AlpacaPaperTrader()
    
    # Run daily update
    success = trader.run_daily_update()
    
    # Show performance
    trader.get_performance_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()