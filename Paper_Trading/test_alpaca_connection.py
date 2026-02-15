from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.common.exceptions import APIError
import config
import sys


def test_connection():
    
    print("\n")
    print("TESTING ALPACA CONNECTION")
    print("\n")
    
    all_passed = True
    
    try:
        print("\n Test 1: Verifying API credentials...")
        
        try:
            trading_client = TradingClient(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                paper=config.PAPER_TRADING
            )
            
            account = trading_client.get_account()
            print(f"   ✓ Connection successful!")
            print(f"   ✓ Account authenticated")
            
        except APIError as e:
            print(f"   ✗ Authentication failed: {e}")
            print(f"\n💡 Common fixes:")
            print(f"   - Check API keys in config.py")
            print(f"   - Verify keys are from paper trading account")
            print(f"   - Check for extra spaces in keys")
            all_passed = False
            return False
        print("\nTest 2: Fetching account information")
        
        print(f"\n    Account Details:")
        print(f"   • Status: {account.status}")
        print(f"   • Account Number: {account.account_number}")
        print(f"   • Trading Blocked: {account.trading_blocked}")
        print(f"   • Account Blocked: {account.account_blocked}")
        
        print(f"\n    Balances:")
        print(f"   • Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   • Cash: ${float(account.cash):,.2f}")
        print(f"   • Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   • Equity: ${float(account.equity):,.2f}")

        if account.trading_blocked:
            print(f"     WARNING: Trading is blocked on this account!")
            all_passed = False
        
        if account.account_blocked:
            print(f"     WARNING: Account is blocked!")
            all_passed = False
        
        if float(account.buying_power) < 1000:
            print(f"     WARNING: Low buying power (${float(account.buying_power):,.2f})")

        print("\n Test 3: Checking market status...")
        
        clock = trading_client.get_clock()
        market_status = "🟢 OPEN" if clock.is_open else "🔴 CLOSED"
        
        print(f"   • Market: {market_status}")
        print(f"   • Current Time: {clock.timestamp}")
        print(f"   • Next Open: {clock.next_open}")
        print(f"   • Next Close: {clock.next_close}")
        
        print("\n Test 4: Testing market data access...")
        
        try:
            data_client = StockHistoricalDataClient(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY
            )
            
            # Test data for validated pairs
            test_symbols = []
            for pair_config in config.VALIDATED_PAIRS.values():
                test_symbols.extend([pair_config['ticker_y'], pair_config['ticker_x']])
            
            if not test_symbols:
                print(f"    No pairs configured for testing")
                test_symbols = ['AAPL']  # Fallback
            
            print(f"   • Testing symbols: {', '.join(test_symbols[:2])}")
            
            request = StockLatestTradeRequest(
                symbol_or_symbols=test_symbols[:2]
            )
            
            trades = data_client.get_stock_latest_trade(request)
            
            for symbol, trade in trades.items():
                print(f"   • {symbol}: ${trade.price:.2f} (Size: {trade.size})")
            
            print(f"    Market data access working!")
            
        except APIError as e:
            if "market is closed" in str(e).lower():
                print(f"    Market is closed, data may be delayed")
                print(f"     This is normal - test again during market hours")
            else:
                print(f"   ✗ Data access error: {e}")
                all_passed = False
        print("\n Test 5: Checking current positions")
        
        positions = trading_client.get_all_positions()
        
        if len(positions) == 0:
            print(f"    No open positions")
            print(f"    This is expected for a new account")
        else:
            print(f"    Found {len(positions)} open position(s):")
            for pos in positions:
                pl = float(pos.unrealized_pl)
                pl_pct = float(pos.unrealized_plpc) * 100
                pl_color = "🟢" if pl >= 0 else "🔴"
                
                print(f"   {pl_color} {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
                print(f"      Current: ${float(pos.current_price):.2f} | P/L: ${pl:.2f} ({pl_pct:+.2f}%)")
        
        print("\n Test 6: Checking order history")
        
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        orders = trading_client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=5
            )
        )
        
        if len(orders) == 0:
            print(f"   • No order history")
            print(f"   This is expected for a new account")
        else:
            print(f"   • Found {len(orders)} recent order(s)")
            for order in orders[:3]:  
                print(f"   • {order.symbol}: {order.side} {order.qty} @ {order.type} - {order.status}")

        print("\n Test 7: Validating configuration")
        
        if config.validate_config():
            print(f"    Configuration is valid")
        else:
            print(f"    Configuration has errors")
            all_passed = False
        
        if config.VALIDATED_PAIRS:
            print(f"\n    Validated Pairs:")
            for name, pair_config in config.VALIDATED_PAIRS.items():
                print(f"   • {name}: {pair_config['ticker_y']}/{pair_config['ticker_x']}")
                print(f"     Position Size: {pair_config['position_size']:.1%}")
                print(f"     Entry Z (Normal): ±{pair_config['entry_z_normal']}")
                print(f"     Exit Z (Normal): ±{pair_config['exit_z_normal']}")
        else:
            print(f"     No pairs configured yet")
        
        print("\n Test 8: Checking risk management settings...")
        
        print(f"   • Initial Capital: ${config.INITIAL_CAPITAL:,.0f}")
        print(f"   • Max Position Size: {config.MAX_POSITION_SIZE:.0%}")
        print(f"   • Max Drawdown: {config.MAX_DRAWDOWN:.0%}")
        print(f"   • Daily Loss Limit: {config.DAILY_LOSS_LIMIT:.0%}")
        print(f"   ✓ Risk management configured")
        
        print("\n" + "="*10)
        
        if all_passed:
            print(" ALL TESTS PASSED!")
            print("\n  Alpaca connection is working correctly")
            
            if not clock.is_open:
                print("\n  Market is currently closed")
                print(f"   Next open: {clock.next_open}")
            
            print("\n")
            print("\n")
            return True
        else:
            print("  SOME TESTS FAILED")
            print("\n" )
            return False
            
    except Exception as e:
        print("\n" + "="*10)
        print(" CONNECTION TEST FAILED")
        print("\n")
        print("\n")
        print(f"\nError: {e}")
        print("\n Troubleshooting Steps:")
        print("   1. Verify API keys in config.py are correct")
        print("   2. Check that keys are from paper trading account")
        print("   3. Ensure no extra spaces in API keys")
        print("   4. Verify internet connection is working")
        print("   5. Check Alpaca status at: https://status.alpaca.markets/")
        print("\n Get your API keys at:")
        print("   https://app.alpaca.markets/paper/dashboard/overview")
        print("\n")
        print("\n")
        
        import traceback
        print("\n Full error trace:")
        traceback.print_exc()
        
        return False


def quick_test():
    try:
        trading_client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=config.PAPER_TRADING
        )
        
        account = trading_client.get_account()
        return True
        
    except Exception as e:
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick connection test...")
        if quick_test():
            print(" Connection OK")
            sys.exit(0)
        else:
            print(" Connection Failed")
            sys.exit(1)
    else:
        success = test_connection()
        sys.exit(0 if success else 1)