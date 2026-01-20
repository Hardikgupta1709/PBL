"""
Compare Ultra-Conservative vs Ultimate Hybrid System
Run this after 90 days of paper trading to see if upgrade is worth it
"""

from Core_Strategy.ultra_conservative_strategy import UltraConservativeSystem, download_data
from hybrid_pairs_trading import UltimateHybridSystem
import pandas as pd

def compare_systems(ticker_y='SO', ticker_x='SRE', 
                   start_date='2020-01-01', end_date='2024-01-01'):
    """
    Compare both systems side-by-side
    """
    
    print("="*80)
    print("SYSTEM COMPARISON: Ultra-Conservative vs Ultimate Hybrid")
    print("="*80)
    
    # Download data
    print("\n📥 Downloading data...")
    stock_y, stock_x, market = download_data(ticker_y, ticker_x, 'SPY', 
                                             start_date, end_date)
    
    # Test 1: Ultra-Conservative System
    print("\n" + "="*80)
    print("TEST 1: ULTRA-CONSERVATIVE SYSTEM")
    print("="*80)
    
    uc_system = UltraConservativeSystem()
    uc_results = uc_system.run_backtest(
        stock_y, stock_x, market,
        train_period=252,
        entry_z_normal=2.0,
        exit_z_normal=0.5,
        entry_z_volatile=2.5,
        exit_z_volatile=0.3,
        min_hold_days=3,
        z_score_window=60
    )
    
    uc_metrics = uc_system.get_performance_metrics()
    uc_trades = uc_system.get_trade_analysis()
    
    # Test 2: Ultimate Hybrid System
    print("\n" + "="*80)
    print("TEST 2: ULTIMATE HYBRID SYSTEM")
    print("="*80)
    
    uh_system = UltimateHybridSystem()
    uh_results = uh_system.run_backtest(
        stock_y, stock_x, market,
        train_period=252,
        entry_z_normal=2.0,
        exit_z_normal=0.5,
        entry_z_volatile=2.5,
        exit_z_volatile=0.3,
        min_hold_days=3,
        min_correlation=0.5,
        use_dynamic_sizing=True,
        use_multi_timeframe=True,
        use_coint_monitoring=True
    )
    
    uh_metrics = uh_system.get_performance_metrics()
    uh_trades = uh_system.get_trade_analysis()
    
    # Comparison
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Ultra-Conservative': {
            'Total Return': uc_metrics['Hybrid Strategy']['Total Return'],
            'Sharpe Ratio': uc_metrics['Hybrid Strategy']['Sharpe Ratio'],
            'Max Drawdown': uc_metrics['Hybrid Strategy']['Max Drawdown'],
            'Win Rate': uc_metrics['Hybrid Strategy']['Win Rate'],
            'Total Trades': uc_metrics['Hybrid Strategy']['Total Trades']
        },
        'Ultimate Hybrid': {
            'Total Return': uh_metrics['Ultimate Strategy']['Total Return'],
            'Sharpe Ratio': uh_metrics['Ultimate Strategy']['Sharpe Ratio'],
            'Max Drawdown': uh_metrics['Ultimate Strategy']['Max Drawdown'],
            'Win Rate': uh_metrics['Ultimate Strategy']['Win Rate'],
            'Total Trades': uh_metrics['Ultimate Strategy']['Total Trades']
        }
    })
    
    print("\n" + comparison.to_string())
    
    # Trade Analysis
    print("\n" + "="*80)
    print("💼 TRADE ANALYSIS")
    print("="*80)
    
    if len(uc_trades) > 0 and len(uh_trades) > 0:
        print(f"\nUltra-Conservative:")
        print(f"  Total Trades: {len(uc_trades)}")
        print(f"  Win Rate: {uc_trades['profitable'].mean()*100:.1f}%")
        print(f"  Avg P&L: {uc_trades['pnl_pct'].mean():.2f}%")
        print(f"  Max Consecutive Losses: {calculate_max_streak(uc_trades)}")
        
        print(f"\nUltimate Hybrid:")
        print(f"  Total Trades: {len(uh_trades)}")
        print(f"  Win Rate: {uh_trades['profitable'].mean()*100:.1f}%")
        print(f"  Avg P&L: {uh_trades['pnl_pct'].mean():.2f}%")
        print(f"  Max Consecutive Losses: {calculate_max_streak(uh_trades)}")
    
    # Recommendation
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION")
    print("="*80)
    
    uc_sharpe = float(uc_metrics['Hybrid Strategy']['Sharpe Ratio'])
    uh_sharpe = float(uh_metrics['Ultimate Strategy']['Sharpe Ratio'])
    
    if uh_sharpe > uc_sharpe * 1.2:  # 20% improvement
        print("\n✅ UPGRADE RECOMMENDED")
        print(f"\nUltimate Hybrid shows {((uh_sharpe/uc_sharpe - 1)*100):.1f}% improvement")
        print("Worth testing in paper trading!")
        
    elif uh_sharpe > uc_sharpe:
        print("\n⚠️ MARGINAL IMPROVEMENT")
        print(f"\nUltimate Hybrid shows {((uh_sharpe/uc_sharpe - 1)*100):.1f}% improvement")
        print("Consider testing, but not urgent")
        
    else:
        print("\n❌ STICK WITH ULTRA-CONSERVATIVE")
        print("\nUltimate Hybrid does not show improvement")
        print("Continue with current system")
    
    return comparison

def calculate_max_streak(trades):
    """Calculate maximum consecutive losses"""
    max_streak = 0
    current_streak = 0
    
    for profitable in trades['profitable']:
        if not profitable:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


if __name__ == "__main__":
    comparison = compare_systems('SO', 'SRE')
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)