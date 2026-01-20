#!/usr/bin/env python3
"""
Paper Trading Runner - Works from anywhere
"""

import sys
import os

# Get project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import what we need
from Paper_Trading.alpaca_paper_trader import AlpacaPaperTrader

def main():
    print("="*80)
    print("PAPER TRADING SYSTEM - DAILY UPDATE")
    print("="*80)
    print()
    
    try:
        trader = AlpacaPaperTrader()
        trader.run_daily_update()
        trader.get_performance_summary()
        
        print("\n" + "="*80)
        print("✅ UPDATE COMPLETE")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED")
        print("="*80)
        print(f"\nError: {e}")
        
        import traceback
        traceback.print_exc()
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
