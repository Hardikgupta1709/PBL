from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.common.exceptions import APIError

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import Paper_Trading.config as config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlpacaPaperTrader:
    
    def __init__(self):
        logger.info("Initializing  Alpaca Paper Trader")
        
        self.trading_client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.PAPER_TRADING
        )
        
        self.data_client = StockHistoricalDataClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY
        )
        
        self.pairs = config.VALIDATED_PAIRS
        self.log_file = config.TRADE_LOG_FILE
        
        # Price cache
        self._price_cache = {}
        self._cache_timeout = 60
        
        logger.info(f"Initialized with {len(self.pairs)} pair(s)")
    
    def get_account_summary(self) -> Dict:
        account = self.trading_client.get_account()
        
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'initial_capital': config.INITIAL_CAPITAL,
            'total_return': (float(account.portfolio_value) - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL,
            'status': account.status,
            'trading_blocked': account.trading_blocked
        }
    
    def get_current_positions(self) -> Dict[str, float]:
        positions = self.trading_client.get_all_positions()
        return {pos.symbol: float(pos.qty) for pos in positions}
    
    def get_batch_prices(self, symbols: list) -> Dict[str, float]:
        request = StockLatestTradeRequest(symbol_or_symbols=symbols)
        trades = self.data_client.get_stock_latest_trade(request)
        
        prices = {}
        for symbol, trade in trades.items():
            price = float(trade.price)
            prices[symbol] = price
            self._price_cache[symbol] = (price, datetime.now())
        
        return prices
    
    def calculate_signals(self, pair_name: str, pair_config: Dict) -> Optional[Dict]:
        ticker_y = pair_config['ticker_y']
        ticker_x = pair_config['ticker_x']
        
        logger.info(f"Calculating signals for {pair_name}")
        
        try:
            from Core_Strategy.conservative_strategy import (
                ConservativeSystem,
                download_data
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.DATA_LOOKBACK_DAYS)
            
            stock_y, stock_x, market = download_data(
                ticker_y, ticker_x, config.BENCHMARK_SYMBOL,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(stock_y) < config.MIN_DATA_POINTS:
                logger.warning(f"Insufficient data: {len(stock_y)} points")
                return None
            
            system = ConservativeSystem()
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
                'timestamp': datetime.now(),
                'confidence': abs(latest_z) / pair_config['entry_z_normal']
            }
            
        except Exception as e:
            logger.error(f"Error calculating signals: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_increment_size(
        self,
        signal_strength: float,
        current_position_pct: float,
        max_position_pct: float,
        account_value: float
    ) -> float:
        # Base increment: 25% of maximum position
        base_increment_pct = max_position_pct * 0.25
        
        # Adjust by signal strength (1.0 = at threshold, 2.0 = 2x threshold)
        strength_multiplier = min(signal_strength / 1.5, 1.0)
        
        increment_pct = base_increment_pct * strength_multiplier
        
        # Calculate new target (not exceeding max)
        new_target_pct = min(
            current_position_pct + increment_pct,
            max_position_pct
        )
        
        # Converting to dollar value (THIS IS THE INCREMENT, not total)
        increment_value = account_value * increment_pct
        
        return increment_value
    
    def execute_order(self, symbol: str, qty: int, side: str) -> Optional[Dict]:
        if qty == 0:
            return None
        
        qty = abs(int(qty))
        
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            tif = TimeInForce[config.TIME_IN_FORCE.upper()]
            
            order_data = MarketOrderRequest(
                symbol=symbol.upper(),
                qty=qty,
                side=order_side,
                time_in_force=tif
            )
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f" Order: {side.upper()} {qty} {symbol} - ID: {order.id}")
            
            return order
            
        except APIError as e:
            logger.error(f" Order failed for {symbol}: {e}")
            return None
    
    def manage_pair_FIXED(
        self,
        pair_name: str,
        pair_config: Dict,
        signal_data: Dict
    ) -> bool:
        ticker_y = pair_config['ticker_y']
        ticker_x = pair_config['ticker_x']
        
        # Getting current state
        current_positions = self.get_current_positions()
        current_y = current_positions.get(ticker_y, 0)
        current_x = current_positions.get(ticker_x, 0)
        
        signal = signal_data['signal']
        hedge_ratio = signal_data['hedge_ratio']
        confidence = signal_data['confidence']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Managing {pair_name} (FIXED VERSION)")
        logger.info(f"Signal: {signal} | Z: {signal_data['z_score']:.2f} | Confidence: {confidence:.2f}")
        logger.info(f"Current: Y={current_y:.0f}, X={current_x:.0f}")
        
        account = self.get_account_summary()
        
        if account['trading_blocked']:
            logger.warning("Trading blocked")
            return False
        
        # Getting prices
        prices = self.get_batch_prices([ticker_y, ticker_x])
        price_y = prices[ticker_y]
        price_x = prices[ticker_x]
        
        logger.info(f"Prices: {ticker_y}=${price_y:.2f}, {ticker_x}=${price_x:.2f}")
        
        # Calculate current position value
        current_value_y = abs(current_y * price_y)
        current_value_x = abs(current_x * price_x)
        current_total_value = current_value_y + current_value_x
        current_position_pct = current_total_value / account['portfolio_value']
        
        logger.info(f"Current position: {current_position_pct:.1%} of portfolio")
        
        # EXIT logic 
        if signal == 0:
            if current_y != 0 or current_x != 0:
                logger.info(f"🚪 EXIT signal - closing positions")
                
                # Close Y
                if abs(current_y) > 0:
                    side_y = 'sell' if current_y > 0 else 'buy'
                    self.execute_order(ticker_y, abs(current_y), side_y)
                
                # Close X
                if abs(current_x) > 0:
                    side_x = 'sell' if current_x > 0 else 'buy'
                    self.execute_order(ticker_x, abs(current_x), side_x)
                
                logger.info(f"{'='*60}\n")
                return True
            else:
                logger.info(f"Already flat")
                return False
        
        # ENTRY logic
        max_position_pct = pair_config['position_size']  
        
        if current_position_pct >= max_position_pct * 0.95:    # Checking for max
            logger.info(f"  At max position ({current_position_pct:.1%}), not adding")
            logger.info(f"{'='*60}\n")
            return False
        
        # Calculating increment 
        increment_value = self.calculate_increment_size(
            signal_strength=confidence,
            current_position_pct=current_position_pct,
            max_position_pct=max_position_pct,
            account_value=account['portfolio_value']
        )
        
        if increment_value < 100:
            logger.info(f"  Increment too small (${increment_value:.0f})")
            logger.info(f"{'='*60}\n")
            return False
        
        logger.info(f" Adding ${increment_value:,.0f} to position (gradual entry)")
        
        # Calculate share quantities for INCREMENT
        if signal == 1:  # LONG spread
            add_y = int(increment_value / price_y)
            add_x = -int(add_y * hedge_ratio)
            action = "ADD LONG"
        else:  # SHORT spread
            add_y = -int(increment_value / price_y)
            add_x = int(abs(add_y) * hedge_ratio)
            action = "ADD SHORT"
        
        logger.info(f"Action: {action}")
        logger.info(f"Adding: Y={add_y:+}, X={add_x:+}")
        
        # Calculate new targets
        target_y = current_y + add_y
        target_x = current_x + add_x
        
        logger.info(f"New targets: Y={target_y}, X={target_x}")
        
        # Execute
        orders_placed = []
        
        if abs(add_y) > 0:
            side_y = 'buy' if add_y > 0 else 'sell'
            order_y = self.execute_order(ticker_y, abs(add_y), side_y)
            if order_y:
                orders_placed.append(order_y)
        
        if abs(add_x) > 0:
            side_x = 'buy' if add_x > 0 else 'sell'
            order_x = self.execute_order(ticker_x, abs(add_x), side_x)
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
        
        logger.info(f"{'='*60}\n")
        
        return len(orders_placed) > 0
    
    def log_trade(self, pair_name: str, signal_data: Dict, qty_y: int, qty_x: int,
                  price_y: float, price_x: float, orders: list):
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
            
            logger.info(f"Trade logged")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def run_daily_update(self) -> bool:
        logger.info(" DAILY UPDATE")
        
        try:
            clock = self.trading_client.get_clock()
            
            if clock.is_open:
                logger.warning("Market open.")
            
            account = self.get_account_summary()
            
            logger.info(f"\n Account:")
            logger.info(f"Portfolio: ${account['portfolio_value']:,.2f}")
            logger.info(f"Cash: ${account['cash']:,.2f}")
            logger.info(f"Return: {account['total_return']:+.2%}")
            
            if account['trading_blocked']:
                logger.error("Trading blocked!")
                return False
            
            # Processing pairs
            success_count = 0
            
            for pair_name, pair_config in self.pairs.items():
                try:
                    logger.info(f"\nProcessing {pair_name}...")
                    
                    signal_data = self.calculate_signals(pair_name, pair_config)
                    
                    if signal_data is None:
                        logger.warning(f"Skipping {pair_name}")
                        continue
                    
                    if self.manage_pair_FIXED(pair_name, pair_config, signal_data):
                        success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.info("\n")
            logger.info("\n")
            logger.info(f" Complete! {success_count}/{len(self.pairs)} pair(s)")
            logger.info("\n")
            logger.info("\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Critical error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    
    if not config.validate_config():
        logger.error("Config validation failed!")
        sys.exit(1)
    
    trader = AlpacaPaperTrader()
    success = trader.run_daily_update()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()