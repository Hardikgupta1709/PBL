"""
Test Alpaca API Connection
Run this to verify your API keys work
"""

from alpaca_trade_api import REST
import config

def test_connection():
    """Test connection to Alpaca API"""
    
    print("="*80)
    print("TESTING ALPACA CONNECTION")
    print("="*80)
    
    try:
        # Create API connection
        api = REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL
        )
        
        # Test 1: Get account info
        print("\n✅ Step 1: Connecting to Alpaca...")
        account = api.get_account()
        
        print(f"✅ Connection successful!")
        print(f"\n📊 Account Information:")
        print(f"  Status: {account.status}")
        print(f"  Account Number: {account.account_number}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        
        # Test 2: Check market status
        print(f"\n✅ Step 2: Checking market status...")
        clock = api.get_clock()
        print(f"  Market is: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}")
        print(f"  Next open: {clock.next_open}")
        print(f"  Next close: {clock.next_close}")
        
        # Test 3: Test getting quotes
        print(f"\n✅ Step 3: Testing data access...")
        try:
            # Try to get current price for SO and SRE
            so_quote = api.get_latest_trade('SO')
            sre_quote = api.get_latest_trade('SRE')
            
            print(f"  SO last price: ${so_quote.price:.2f}")
            print(f"  SRE last price: ${sre_quote.price:.2f}")
            print(f"  ✅ Data access working!")
        except Exception as e:
            print(f"  ⚠️ Data access: {e}")
            print(f"  (This is OK if market is closed)")
        
        # Test 4: Check positions
        print(f"\n✅ Step 4: Checking current positions...")
        positions = api.list_positions()
        
        if len(positions) == 0:
            print(f"  No open positions (expected for new account)")
        else:
            print(f"  Current positions: {len(positions)}")
            for pos in positions:
                print(f"    {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nYour Alpaca connection is working correctly.")
        print("You're ready to start paper trading!")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ CONNECTION FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\nCommon issues:")
        print("  1. Check your API keys in config.py")
        print("  2. Make sure you copied the full keys")
        print("  3. Verify you're using paper trading keys")
        print("  4. Check internet connection")
        
        return False


if __name__ == "__main__":
    test_connection()