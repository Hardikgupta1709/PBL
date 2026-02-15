import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import Paper_Trading.config as config
from Paper_Trading.test_alpaca_connection import test_connection, quick_test
from Paper_Trading.alpaca_paper_trader import AlpacaPaperTrader
from Paper_Trading.Monitoring_Dashboard import TradingMonitor


def run_connection_test(quick: bool = False) -> bool:
    print("\nTesting Alpaca Connection")
    
    if quick:
        return quick_test()
    else:
        return test_connection()


def run_daily_trading() -> bool:
    print("\n Running Daily Paper Trading Update")
    
    try:
        if not config.validate_config():
            print(" Configuration validation failed!")
            return False

        trader = AlpacaPaperTrader()
        
        success = trader.run_daily_update()
        
        if hasattr(trader, 'get_performance_summary'):
            trader.get_performance_summary()
        else:
            print("\n Performance Summary:")
            account = trader.get_account_summary()
            print(f"  Portfolio: ${account['portfolio_value']:,.2f}")
            print(f"  Return: {account['total_return']:+.2%}")
        
        return success
        
    except Exception as e:
        print(f" Error in daily trading: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_monitoring() -> bool:
    print("\n Generating Monitoring Report")
    
    try:
        monitor = TradingMonitor()
        monitor.generate_report()
        try:
            monitor.plot_performance()
        except Exception as e:
            print(f" Could not generate charts: {e}")
        
        return True
        
    except Exception as e:
        print(f" Error in monitoring: {e}")
        return False


def run_full_suite():
    
    print("\n")
    print("\n")
    print(" PAPER TRADING SYSTEM - FULL ")
    print("\n")
    print("\n")
    
    print("\n")
    print("\n")
    print("STEP 1: CONNECTION TEST")
    print("\n")
    print("\n")
    
    if not run_connection_test(quick=False):
        print("\n Connection test failed. Please fix issues before continuing.")
        return False
    
    print("\n")
    print("\n")
    print("STEP 2: DAILY TRADING UPDATE")
    print("\n")
    print("\n")
    
    if not run_daily_trading():
        print("\n Trading update had issues. Check logs for details.")
    
    print("\n")
    print("\n")
    print("STEP 3: MONITORING DASHBOARD")
    print("\n")
    print("\n")
    
    run_monitoring()
    
    print("\n")
    print(" FULL RUN COMPLETE")
    print("\n")
    print("\n")
    
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
    
    print("\n")
    print("\n")
    print(" PAPER TRADING SYSTEM")
    print("\n")
    print("\n")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Paper Trading' if config.PAPER_TRADING else 'LIVE TRADING '}")
    print("\n")
    print("\n")
    
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
        print("\n Interactive Mode")
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
                    print("\n Goodbye!")
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
                    print(" Invalid choice. Please enter 0-5.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

