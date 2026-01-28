<<<<<<< HEAD
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
=======
"""
Main Runner Script for Paper Trading System
Optimized for daily execution with scheduling support
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import Paper_Trading.config as config
from Paper_Trading.test_alpaca_connection import test_connection, quick_test
from Paper_Trading.alpaca_paper_trader import AlpacaPaperTrader
from Paper_Trading.Monitoring_Dashboard import TradingMonitor


def run_connection_test(quick: bool = False) -> bool:
    """
    Test Alpaca connection
    
    Args:
        quick: Whether to run quick test
        
    Returns:
        True if connection successful
    """
    print("\n🔌 Testing Alpaca Connection...")
    
    if quick:
        return quick_test()
    else:
        return test_connection()


def run_daily_trading() -> bool:
    """
    Run daily paper trading update
    
    Returns:
        True if successful
    """
    print("\n🚀 Running Daily Paper Trading Update...")
    
    try:
        # Validate config
        if not config.validate_config():
            print("❌ Configuration validation failed!")
            return False
        
        # Create trader
        trader = AlpacaPaperTrader()
        
        # Run update
        success = trader.run_daily_update()
        
        # Show performance
        trader.get_performance_summary()
        
        return success
        
    except Exception as e:
        print(f"❌ Error in daily trading: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_monitoring() -> bool:
    """
    Run monitoring dashboard
    
    Returns:
        True if successful
    """
    print("\n📊 Generating Monitoring Report...")
    
    try:
        monitor = TradingMonitor()
        monitor.generate_report()
        
        # Try to generate charts
        try:
            monitor.plot_performance()
        except Exception as e:
            print(f"⚠️  Could not generate charts: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in monitoring: {e}")
        return False


def run_full_suite():
    """Run complete test suite"""
    
    print("="*80)
    print("🎯 PAPER TRADING SYSTEM - FULL SUITE")
    print("="*80)
    
    # Step 1: Test connection
    print("\n" + "="*80)
    print("STEP 1: CONNECTION TEST")
    print("="*80)
    
    if not run_connection_test(quick=False):
        print("\n❌ Connection test failed. Please fix issues before continuing.")
        return False
    
    # Step 2: Run trading
    print("\n" + "="*80)
    print("STEP 2: DAILY TRADING UPDATE")
    print("="*80)
    
    if not run_daily_trading():
        print("\n⚠️  Trading update had issues. Check logs for details.")
    
    # Step 3: Monitoring
    print("\n" + "="*80)
    print("STEP 3: MONITORING DASHBOARD")
    print("="*80)
    
    run_monitoring()
    
    print("\n" + "="*80)
    print("✅ FULL SUITE COMPLETE")
    print("="*80)
    
    return True


def main():
    """Main entry point with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description='Paper Trading System - Optimized for Alpaca',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_paper_trading.py --test           # Test connection only
  python run_paper_trading.py --trade          # Run daily trading
  python run_paper_trading.py --monitor        # Generate monitoring report
  python run_paper_trading.py --full           # Run complete suite
  python run_paper_trading.py                  # Interactive mode

Daily Usage:
  Schedule this to run daily after market close (4:30 PM ET):
  python run_paper_trading.py --trade
        """
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test Alpaca API connection'
    )
    
    parser.add_argument(
        '--quick-test', '-q',
        action='store_true',
        help='Quick connection test (less verbose)'
    )
    
    parser.add_argument(
        '--trade',
        action='store_true',
        help='Run daily paper trading update'
    )
    
    parser.add_argument(
        '--monitor', '-m',
        action='store_true',
        help='Generate monitoring dashboard'
    )
    
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run full test suite'
    )
    
    parser.add_argument(
        '--config-summary',
        action='store_true',
        help='Print configuration summary'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("📈 PAPER TRADING SYSTEM")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Paper Trading' if config.PAPER_TRADING else 'LIVE TRADING ⚠️'}")
    print("="*80)
    
    success = True
    
    # Handle commands
    if args.config_summary:
        config.print_config_summary()
    
    elif args.test:
        success = run_connection_test(quick=False)
    
    elif args.quick_test:
        success = run_connection_test(quick=True)
    
    elif args.trade:
        success = run_daily_trading()
    
    elif args.monitor:
        success = run_monitoring()
    
    elif args.full:
        success = run_full_suite()
    
    else:
        # Interactive mode
        print("\n🤖 Interactive Mode")
        print("\nWhat would you like to do?")
        print("  1. Test connection")
        print("  2. Run daily trading")
        print("  3. View monitoring dashboard")
        print("  4. Run full suite")
        print("  5. View configuration")
        print("  0. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (0-5): ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye!")
                    break
                    
                elif choice == '1':
                    run_connection_test(quick=False)
                    
                elif choice == '2':
                    run_daily_trading()
                    
                elif choice == '3':
                    run_monitoring()
                    
                elif choice == '4':
                    run_full_suite()
                    break
                    
                elif choice == '5':
                    config.print_config_summary()
                    
                else:
                    print("❌ Invalid choice. Please enter 0-5.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
>>>>>>> 3db96387 (Updated Execution of Strategy)
