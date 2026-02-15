import sys
sys.path.insert(0, '/Users/hardik/PBL_Run')

from alpaca.trading.client import TradingClient
import Paper_Trading.config as config

def emergency_exit():
    
    print("="*10)
    print(" EMERGENCY EXIT - CLOSING ALL POSITIONS")
    print("="*10)
    
    client = TradingClient(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        paper=config.PAPER_TRADING
    )
    
    # Getting positions
    positions = client.get_all_positions()
    
    if len(positions) == 0:
        print(" No positions to close")
        return
    
    print(f"\n Current Positions:")
    total_pl = 0
    for pos in positions:
        pl = float(pos.unrealized_pl)
        total_pl += pl
        print(f"  {pos.symbol}: {pos.qty} shares | P/L: ${pl:,.2f}")
    
    print(f"\n Total Unrealized P/L: ${total_pl:,.2f}")
    response = input("Type 'YES' to confirm: ")
    
    if response != 'YES':
        print("Exit cancelled")
        return
    print("\n Closing positions...")
    
    for pos in positions:
        try:
            client.close_position(pos.symbol)
            print(f"   Closed {pos.symbol}")
        except Exception as e:
            print(f"   Error closing {pos.symbol}: {e}")

    import time
    time.sleep(2)
    
    remaining = client.get_all_positions()
    
    if len(remaining) == 0:
        print("\n ALL POSITIONS CLOSED!")
    else:
        print(f"\n  {len(remaining)} positions still open")
        for pos in remaining:
            print(f"  {pos.symbol}: {pos.qty}")
    
    account = client.get_account()
    pv = float(account.portfolio_value)
    loss = (pv - config.INITIAL_CAPITAL)
    loss_pct = (loss / config.INITIAL_CAPITAL) * 100
    
    print(f"\n Final Status:")
    print(f"  Portfolio Value: ${pv:,.2f}")
    print(f"  Cash: ${float(account.cash):,.2f}")
    print(f"  Total Loss: ${loss:,.2f} ({loss_pct:+.2f}%)")
    
    print("\n" + "="*10)
    print("POSITIONS CLOSED")
    print("="*10)

if __name__ == "__main__":
    emergency_exit()