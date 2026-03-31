import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import Paper_Trading.config as config
from Paper_Trading.test_alpaca_connection import test_connection, quick_test
from Paper_Trading.alpaca_paper_trader import AlpacaPaperTrader
from Paper_Trading.Monitoring_Dashboard import TradingMonitor
from Paper_Trading.pair_rotation import PairRotationManager, get_rotated_pairs


def run_connection_test(quick: bool = False) -> bool:
    print("\nTesting Alpaca Connection")
    
    if quick:
        return quick_test()
    else:
        return test_connection()


def run_pair_rotation(top_k: int = 3, force: bool = False) -> bool:
    """Run automatic pair rotation — scan universe, select healthiest pairs."""
    print("\n Running Automatic Pair Rotation")
    
    try:
        manager = PairRotationManager(
            top_k=top_k,
            min_health_score=config.ROTATION_MIN_HEALTH_SCORE,
            min_healthy_pct=config.ROTATION_MIN_HEALTHY_PCT,
            rotation_interval_days=config.ROTATION_INTERVAL_DAYS,
            health_window=config.HEALTH_LOOKBACK_WINDOW,
            adf_pvalue=config.HEALTH_ADF_PVALUE,
            hurst_max=config.HEALTH_HURST_MAX,
            coint_pvalue=config.HEALTH_COINT_PVALUE,
            max_total_exposure=config.ROTATION_MAX_TOTAL_EXPOSURE,
            universe_mode=config.ROTATION_UNIVERSE_MODE,
        )
        result = manager.run_rotation(force=force, verbose=True)
        
        if result is None:
            print("  Rotation not needed yet (using existing pairs).")
            if manager.current_pairs:
                print(f"  Active pairs: {', '.join(manager.current_pairs)}")
            return True
        
        # Update config.VALIDATED_PAIRS with new pairs
        scan = manager.scan_universe(verbose=False)
        new_config = manager.build_validated_pairs_config(
            result.selected_pairs, scan
        )
        config.VALIDATED_PAIRS = new_config
        
        print(f"\n  Config updated with {len(new_config)} pairs:")
        for name, cfg in new_config.items():
            print(f"    {name}: {cfg['ticker_y']}/{cfg['ticker_x']} "
                  f"(health={cfg.get('health_score', 0):.3f})")
        
        return True
        
    except Exception as e:
        print(f" Error in pair rotation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_daily_trading(with_rotation: bool = True, top_k: int = 3) -> bool:
    print("\n Running Daily Paper Trading Update")
    
    try:
        # Step 0: Check if pair rotation is needed
        if with_rotation:
            manager = PairRotationManager(
                top_k=top_k,
                min_health_score=config.ROTATION_MIN_HEALTH_SCORE,
                min_healthy_pct=config.ROTATION_MIN_HEALTHY_PCT,
                rotation_interval_days=config.ROTATION_INTERVAL_DAYS,
                health_window=config.HEALTH_LOOKBACK_WINDOW,
                adf_pvalue=config.HEALTH_ADF_PVALUE,
                hurst_max=config.HEALTH_HURST_MAX,
                coint_pvalue=config.HEALTH_COINT_PVALUE,
                max_total_exposure=config.ROTATION_MAX_TOTAL_EXPOSURE,
                universe_mode=config.ROTATION_UNIVERSE_MODE,
            )
            if manager.needs_rotation():
                print("  Monthly rotation check triggered...")
                run_pair_rotation(top_k=top_k)
            elif manager.current_pairs:
                # Load rotated pairs into config
                scan = manager.scan_universe(verbose=False)
                new_config = manager.build_validated_pairs_config(
                    manager.current_pairs, scan
                )
                config.VALIDATED_PAIRS = new_config
                print(f"  Using rotated pairs: {', '.join(manager.current_pairs)}")
        
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
        '--rotate',
        action='store_true',
        help='Run pair rotation (scan universe, select healthiest)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=config.ROTATION_TOP_K,
        help='Number of top pairs to trade (default: 3)'
    )
    
    parser.add_argument(
        '--force-rotate',
        action='store_true',
        help='Force pair rotation regardless of interval'
    )
    
    parser.add_argument(
        '--no-rotation',
        action='store_true',
        help='Skip pair rotation, use config pairs only'
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
    
    elif args.rotate:
        success = run_pair_rotation(top_k=args.top_k, force=args.force_rotate)
    
    elif args.trade:
        success = run_daily_trading(
            with_rotation=not args.no_rotation,
            top_k=args.top_k,
        )
    
    elif args.monitor:
        success = run_monitoring()
    
    elif args.full:
        success = run_full_suite()
    
    else:
        print("\n Interactive Mode")
        print("  1. Test connection")
        print("  2. Run daily trading (with rotation)")
        print("  3. View monitoring dashboard")
        print("  4. Run full suite")
        print("  5. View configuration")
        print("  6. Force pair rotation NOW")
        print("  7. Scan-only (preview rankings)")
        print("  0. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (0-7): ").strip()
                
                if choice == '0':
                    print("\n Goodbye!")
                    break
                    
                elif choice == '1':
                    run_connection_test(quick=False)
                    
                elif choice == '2':
                    run_daily_trading(with_rotation=True)
                    
                elif choice == '3':
                    run_monitoring()
                    
                elif choice == '4':
                    run_full_suite()
                    break
                    
                elif choice == '5':
                    config.print_config_summary()
                
                elif choice == '6':
                    run_pair_rotation(top_k=3, force=True)
                
                elif choice == '7':
                    manager = PairRotationManager(top_k=3)
                    scan = manager.scan_universe(verbose=True)
                    rankings = manager.rank_pairs(scan)
                    print(f"\n  Rankings (top-{manager.top_k} selected):")
                    for i, (name, score, elig) in enumerate(rankings, 1):
                        marker = ' <-- ACTIVE' if i <= manager.top_k and elig else ''
                        status = 'eligible' if elig else 'excluded'
                        print(f"  {i}. {name:<15} score={score:.3f}  {status}{marker}")
                    
                else:
                    print(" Invalid choice. Please enter 0-7.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

